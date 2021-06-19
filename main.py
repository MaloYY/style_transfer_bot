import logging
import os

from aiogram import Bot, types, executor
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.dispatcher import Dispatcher
from aiogram.dispatcher.webhook import SendMessage
from aiogram.utils.executor import start_webhook


API_TOKEN = str(os.getenv('BOT_TOKEN'))
HEROKU_APP_NAME = str(os.getenv('HEROKU_APP_NAME'))

# webhook settings
WEBHOOK_HOST = f'https://{HEROKU_APP_NAME}.herokuapp.com'
WEBHOOK_PATH = f'/{API_TOKEN}'
WEBHOOK_URL = f'{WEBHOOK_HOST}{WEBHOOK_PATH}'

# webserver settings
WEBAPP_HOST = 'localhost'  # or ip
WEBAPP_PORT = 3001

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    """Отправляет приветственное сообщение и помощь по боту"""
    await message.answer('Hello ;)')

@dp.message_handler()
async def echo(message: types.Message):
    logging.warning(f'Recieved a message from {message.from_user}')
    await bot.send_message(message.chat.id, message.text)


async def on_startup(dp):
    logging.warning(
        'Starting connection. ')
    await bot.set_webhook(WEBHOOK_URL,drop_pending_updates=True)


async def on_shutdown(dp):
    logging.warning('Shutting down..')

    # insert code here to run it before shutdown

    # Remove webhook (not acceptable in some cases)
    await bot.delete_webhook()

    # Close DB connection (if used)
    await dp.storage.close()
    await dp.storage.wait_closed()

    logging.warning('Bye! Shutting down webhook connection')


if __name__ == '__main__':
    start_webhook(
        dispatcher=dp,
        webhook_path=WEBHOOK_PATH,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        skip_updates=True,
        host=WEBAPP_HOST,
        port=WEBAPP_PORT,
    )
    #executor.start_polling(dp, skip_updates=True) # delete