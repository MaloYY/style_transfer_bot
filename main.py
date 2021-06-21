import logging
import os
import asyncio

from aiogram import Bot, types, executor
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.dispatcher import Dispatcher
from aiogram.dispatcher.webhook import SendMessage
from aiogram.utils.executor import start_webhook, start_polling

from FaceGAN import FaceGAN

# Easier to test it with pooling
MODE = 'DEPL'  # 'LOCAL'

# configuration
if MODE == 'DEPL':
    API_TOKEN = str(os.getenv('BOT_TOKEN'))
    HEROKU_APP_NAME = str(os.getenv('HEROKU_APP_NAME'))

    # webhook settings
    WEBHOOK_HOST = f'https://{HEROKU_APP_NAME}.herokuapp.com'
    WEBHOOK_PATH = f'/webhook/{API_TOKEN}'
    WEBHOOK_URL = f'{WEBHOOK_HOST}{WEBHOOK_PATH}'

    # webserver settings
    WEBAPP_HOST = '0.0.0.0'  # or ip
    WEBAPP_PORT = int(os.getenv("PORT"))

    bot = Bot(token=API_TOKEN)
    bot.set_webhook(WEBHOOK_URL)
    dp = Dispatcher(bot)
    dp.middleware.setup(LoggingMiddleware())
else:
    API_TOKEN = str(os.getenv('BOT_TOKEN'))
    HEROKU_APP_NAME = str(os.getenv('HEROKU_APP_NAME'))

    # webhook settings
    WEBHOOK_HOST = f'https://{HEROKU_APP_NAME}.herokuapp.com'
    WEBHOOK_PATH = f'/webhook/{API_TOKEN}'
    WEBHOOK_URL = f'{WEBHOOK_HOST}{WEBHOOK_PATH}'

    # webserver settings
    WEBAPP_HOST = '0.0.0.0'  # or ip
    WEBAPP_PORT = 3001

    bot = Bot(token=API_TOKEN)
    dp = Dispatcher(bot)
    dp.middleware.setup(LoggingMiddleware())

logging.basicConfig(level=logging.INFO)


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.answer('Привет, я генерирую лица, у меня есть следующие команды:\n\
/start, /help - вызвать это меню.\n\
/generate - сгенерировать случайное лицо.\n')


@dp.message_handler(commands=['generate'])
async def generate(message: types.Message):
    generator = FaceGAN()
    logging.debug('Generating')
    await generator.get_image()
    if os.path.isfile(f'images/fake.jpg'):
        await bot.send_photo(chat_id=message.from_user.id, photo=open('images/fake.jpg', 'rb'))
    else:
        await message.answer("Didn't find the result")


async def on_startup(dp):
    logging.warning('Starting connection.')
    if MODE == 'DEPL':
        await bot.set_webhook(WEBHOOK_URL, drop_pending_updates=True)


async def on_shutdown(dp):
    logging.warning('Shutting down..')

    # Remove webhook (not acceptable in some cases)
    if MODE == 'DEPL':
        await bot.delete_webhook()

    # Close DB connection (if used)
    await dp.storage.close()
    await dp.storage.wait_closed()

    logging.warning('Bye! Shutting down webhook connection')


def start():
    if MODE == 'DEPL':
        start_webhook(
            dispatcher=dp,
            webhook_path=WEBHOOK_PATH,
            on_startup=on_startup,
            on_shutdown=on_shutdown,
            skip_updates=True,
            host=WEBAPP_HOST,
            port=WEBAPP_PORT,
        )
    else:
        start_polling(
            dispatcher=dp,
            skip_updates=True
        )


if __name__ == '__main__':
    start()
