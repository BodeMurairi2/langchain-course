#!/usr/bin/env python3

import os
from dotenv import load_dotenv
import smtplib
import ssl
from email.message import EmailMessage
from langchain.tools import tool

load_dotenv()

@tool
def send_email(sender_email: str, receiver: str, subject: str, body: str):
    """
    Sends an email using Gmail SMTP.
    """

    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = receiver
    msg["Subject"] = subject
    msg.set_content(body)

    context = ssl.create_default_context()
    gmail_password = os.getenv("GOOGLE_GMAIL_PASSWORD")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
            smtp.login(
                user=sender_email,
                password=gmail_password
            )
            smtp.send_message(msg)

        return f"Email sent successfully to {receiver}"

    except Exception as error:
        print("SMTP ERROR:", error)
        return f"Error sending email:\n{error}"
