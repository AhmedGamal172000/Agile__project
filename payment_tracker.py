"""Telegram payment tracker bot.

This module mirrors the provided n8n workflow in Python:
- Listens for Telegram messages from allowed user IDs.
- Uses OpenAI to extract payment details (amount, category, date, description).
- Maps the result to the correct Google Sheets cell based on date and category.
- Writes the amount to the sheet and replies with a confirmation message.

Configuration is driven by environment variables so the bot can run in different
hosts without code changes.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI
from telegram import Update
from telegram.ext import (Application, CallbackContext, CommandHandler,
                          MessageHandler, filters)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CATEGORY_COLUMN_MAP: Dict[str, str] = {
    "Fuel": "C",
    "Food": "D",
    "Ahwa": "E",
    "Cafe": "F",
    "Coffee machine": "G",
    "Bills": "H",
    "Car Parking & payments": "I",
    "Car Maintenance": "J",
    "Entertainment": "K",
    "others": "L",
    "INCOME": "N",
}

SYSTEM_PROMPT = (
    "You are a payment tracking assistant. Extract payment amount and category from "
    "the message. Categories: Fuel, Food, Ahwa, Cafe, Coffee machine, Bills, Car Parking "
    "& payments, Car Maintenance, Entertainment, others. Return JSON with: amount "
    "(number), category (string), date (MM-DD-YYYY), description (string). Use the "
    "provided current date."
)


@dataclass
class PaymentRecord:
    amount: float
    category: str
    date: str
    description: str

    def date_object(self) -> dt.date:
        try:
            parsed = dt.datetime.strptime(self.date, "%m-%d-%Y").date()
        except ValueError:
            parsed = dt.datetime.strptime(self.date, "%m/%d/%Y").date()
        return parsed

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "PaymentRecord":
        return cls(
            amount=float(payload["amount"]),
            category=str(payload["category"]),
            date=str(payload["date"]),
            description=str(payload.get("description", "")),
        )


class PaymentTracker:
    def __init__(
        self,
        telegram_token: str,
        openai_client: OpenAI,
        google_client: gspread.Client,
        spreadsheet_id: str,
        allowed_user_ids: Optional[Iterable[int]] = None,
        model: str = "gpt-4.1-mini",
    ) -> None:
        self.telegram_token = telegram_token
        self.openai_client = openai_client
        self.google_client = google_client
        self.spreadsheet_id = spreadsheet_id
        self.allowed_user_ids = set(int(uid) for uid in allowed_user_ids or [])
        self.model = model

    # ------------------ Telegram wiring ------------------
    def build_application(self) -> Application:
        app = Application.builder().token(self.telegram_token).build()
        app.add_handler(CommandHandler("start", self.handle_start))
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self.handle_message))
        return app

    async def handle_start(self, update: Update, context: CallbackContext) -> None:
        await update.message.reply_text("Send a payment note (e.g., 'Paid 50 for fuel').")

    async def handle_message(self, update: Update, context: CallbackContext) -> None:
        if not update.message:
            return

        user_id = update.message.from_user.id if update.message.from_user else None
        if self.allowed_user_ids and user_id not in self.allowed_user_ids:
            await update.message.reply_text("🚫 Unauthorized user.")
            return

        message_text = update.message.text or ""
        logger.info("Processing message from %s: %s", user_id, message_text)

        try:
            record = self.extract_payment_record(message_text)
            sheet_name, cell_range = self.calculate_cell(record)
            self.write_amount(sheet_name, cell_range, record.amount)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to process message")
            await update.message.reply_text(f"❌ Could not record payment: {exc}")
            return

        confirmation = (
            "✅ Payment recorded!\n\n"
            f"Amount: {record.amount}\n"
            f"Category: {record.category}\n"
            f"Sheet: {sheet_name}\n"
            f"Cell: {cell_range}\n"
            f"Description: {record.description}"
        )
        await update.message.reply_text(confirmation)

    # ------------------ Extraction & mapping ------------------
    def extract_payment_record(self, message: str) -> PaymentRecord:
        today = dt.datetime.utcnow().strftime("%m/%d/%Y")
        user_prompt = f"Message: {message}\n\nCurrent Date: {today}"
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("OpenAI returned an empty response")
        payload = json.loads(content)
        logger.debug("OpenAI response: %s", payload)
        return PaymentRecord.from_json(payload)

    def calculate_cell(self, record: PaymentRecord) -> tuple[str, str]:
        date_obj = record.date_object()
        sheet_name = f"{date_obj.month}-{date_obj.year}"
        row = date_obj.day + 1  # Header is row 1
        column = CATEGORY_COLUMN_MAP.get(record.category, CATEGORY_COLUMN_MAP["others"])
        return sheet_name, f"{column}{row}"

    # ------------------ Google Sheets ------------------
    def write_amount(self, sheet_name: str, cell: str, amount: float) -> None:
        spreadsheet = self.google_client.open_by_key(self.spreadsheet_id)
        try:
            worksheet = spreadsheet.worksheet(sheet_name)
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=100, cols=20)
        worksheet.update_acell(cell, amount)


# ------------------ Factories ------------------
def build_google_client(service_account_file: str) -> gspread.Client:
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    credentials = Credentials.from_service_account_file(service_account_file, scopes=scopes)
    return gspread.authorize(credentials)


def build_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def load_allowed_user_ids(raw: str) -> Iterable[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    telegram_token = os.environ["TELEGRAM_BOT_TOKEN"]
    openai_api_key = os.environ["OPENAI_API_KEY"]
    spreadsheet_id = os.environ["SPREADSHEET_ID"]
    service_account_file = os.environ["GOOGLE_SERVICE_ACCOUNT_FILE"]
    allowed_users = load_allowed_user_ids(os.environ.get("TELEGRAM_ALLOWED_USER_IDS", ""))

    tracker = PaymentTracker(
        telegram_token=telegram_token,
        openai_client=build_openai_client(openai_api_key),
        google_client=build_google_client(service_account_file),
        spreadsheet_id=spreadsheet_id,
        allowed_user_ids=allowed_users,
    )

    app = tracker.build_application()
    logger.info("Payment tracker bot started and listening for messages...")
    app.run_polling()


if __name__ == "__main__":
    main()
