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