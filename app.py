import os
import sys
from distutils.util import strtobool

import boto3
from dotenv import load_dotenv
from flask import Flask, abort, request
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores.pgvector import PGVector
from linebot import LineBotApi, WebhookHandler, WebhookParser
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from opencc import OpenCC

import src.constants as const
import src.utils as utils
from src.utils import MaxPoolingEmbeddings, PathHelper, get_logger

import threading
from linebot.models import (
    FlexSendMessage,
    BubbleContainer,
    BoxComponent,
    TextComponent,
    LoadingComponent
)

# logger, env and const
logger = get_logger(__name__)
dotenv_path = PathHelper.root_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)
encoding_model_name = const.ENCODING_MODEL_NAME

# get channel_secret and channel_access_token from your environment variable
channel_secret = os.getenv("CHANNEL_SECRET", None)
channel_access_token = os.getenv("CHANNEL_ACCESS_TOKEN", None)
if channel_secret is None:
    print("Specify LINE_CHANNEL_SECRET as environment variable.")
    sys.exit(1)
if channel_access_token is None:
    print("Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.")
    sys.exit(1)

# multi-lingual support
support_multilingual = strtobool(os.getenv("SUPPORT_MULTILINGUAL", "False"))

#
def sync_vector_db_from_s3():
    """download from s3 if folder db/{xxx} is not exist"""
    s3r = boto3.resource(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION_NAME"),
    )
    s3_bucket = s3r.Bucket(const.S3_BUCKET_NAME)
    
    if not os.path.exists(PathHelper.db_dir / const.CHROMA_DB):
        logger.info("downloading db from s3")
        for obj in s3_bucket.objects.filter(Prefix=f"{const.DB}/{const.CHROMA_DB}"):
            if not os.path.exists(os.path.dirname(obj.key)):
                os.makedirs(os.path.dirname(obj.key))
            logger.info(f"download file: {obj.key}")
            s3_bucket.download_file(obj.key, obj.key)

# configure_retriever
def configure_retriever():
    logger.info("configuring retriever")
    embeddings = MaxPoolingEmbeddings(
        api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        model_name=encoding_model_name,
    )

    # vectordb = Chroma(
    #     persist_directory=str(PathHelper.db_dir / const.CHROMA_DB),
    #     embedding_function=embeddings,
    # )
    vectordb = PGVector(
        collection_name=const.COLLECTION_NAME,
        connection_string=utils.get_connection_string(),
        embedding_function=embeddings,
    )
    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": const.N_DOCS}
    )

    logger.info("configuring retriever done")
    return retriever


# create app
app = Flask(__name__)
line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)
parser = WebhookParser(channel_secret)

# initialize agent and retriever
llm = ChatOpenAI(
    model_name=const.CHAT_GPT_MODEL_NAME,
    temperature=0,
    streaming=True,
)

# configure retriever
retriever = configure_retriever()

# create converter (simple chinese to traditional chinese)
s2t_converter = OpenCC("s2t")

# translate and comprehend
translate = boto3.client(
    "translate",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION_NAME"),
)
comprehend = boto3.client(
    "comprehend",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION_NAME"),
)


# create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="question",
    output_key="answer",
)

# use PromptTemplate to generate prompts
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
    verbose=True,
    return_source_documents=True,
)


def create_typing_bubble():
    """創建打字中的 Flex Message"""
    return FlexSendMessage(
        alt_text="思考中...",
        contents=BubbleContainer(
            size="micro",
            body=BoxComponent(
                layout="horizontal",
                contents=[
                    TextComponent(
                        text="思考中",
                        size="sm",
                        color="#8c8c8c",
                        flex=0
                    ),
                    BoxComponent(
                        layout="horizontal",
                        contents=[
                            TextComponent(
                                text=".",
                                size="sm",
                                color="#8c8c8c"
                            ),
                            TextComponent(
                                text=".",
                                size="sm",
                                color="#8c8c8c"
                            ),
                            TextComponent(
                                text=".",
                                size="sm",
                                color="#8c8c8c"
                            )
                        ],
                        flex=0,
                        spacing="none"
                    )
                ],
                spacing="sm",
                alignItems="center"
            )
        )
    )

def process_message_async(event):
    try:
        # 發送打字狀態
        typing_message = create_typing_bubble()
        line_bot_api.push_message(
            event.source.user_id,
            typing_message
        )

        # 處理實際回答
        question = event.message.text.strip()
        
        # 初始化 response 為 None
        response = None
        
        if question.startswith("/清除") or question.lower().startswith("/clear"):
            memory.clear()
            answer = "歷史訊息清除成功"
        elif (
            question.startswith("/教學")
            or question.startswith("/指令")
            or question.startswith("/說明")
            or question.startswith("/操作說明")
            or question.lower().startswith("/instruction")
            or question.lower().startswith("/help")
        ):
            answer = "指令：\n/清除 or /clear\n👉 當 Bot 開始鬼打牆，可清除歷史訊息來重置"
        else:
            try:
                if support_multilingual:
                    question_lang_obj = comprehend.detect_dominant_language(Text=question)
                    question_lang = question_lang_obj["Languages"][0]["LanguageCode"]
                else:
                    question_lang = const.DEFAULT_LANG
                
                logger.info(f"Processing question in language: {question_lang}")
                
                response = qa_chain({"question": question})
                answer = response["answer"]
                answer = s2t_converter.convert(answer)
                
                if (question_lang != const.DEFAULT_LANG) and support_multilingual:
                    answer_translated = translate.translate_text(
                        Text=answer,
                        SourceLanguageCode=const.DEFAULT_LANG,
                        TargetLanguageCode=question_lang,
                    )
                    answer = answer_translated["TranslatedText"]

                # 只在有 response 時添加參考視頻
                if response and "source_documents" in response:
                    ref_video_template = ""
                    for i in range(min(const.N_SOURCE_DOCS, len(response["source_documents"]))):
                        most_related_doc = response["source_documents"][i]
                        most_related_video_id = most_related_doc.metadata["video_id"]
                        url = f"https://www.youtube.com/watch?v={most_related_video_id}"
                        ref_video_template = f"{ref_video_template}\n{url}"
                    
                    answer = f"{answer}\n\nSource: {ref_video_template}"

            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                raise  # 重新拋出異常以觸發外層錯誤處理

        # 發送實際回答
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=answer)
        )

    except Exception as e:
        logger.error(f"Error in process_message_async: {str(e)}")
        try:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="抱歉，處理您的訊息時發生錯誤。請稍後再試。")
            )
        except Exception as reply_error:
            logger.error(f"Error sending error message: {str(reply_error)}")

# create handlers
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
        return "OK", 200
    except Exception as e:
        logger.error(f"Callback error: {str(e)}")
        abort(400)


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    thread = threading.Thread(target=process_message_async, args=(event,))
    thread.daemon = True  # 設置為守護線程
    thread.start()

if __name__ == "__main__":
    # sync_vector_db_from_s3()  # run once
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

    logger.info("app started")
