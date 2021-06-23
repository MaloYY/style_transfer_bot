import logging
import os
import threading
import asyncio

from aiogram import Bot, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.dispatcher import Dispatcher, FSMContext
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher.filters import Text
from aiogram.utils.executor import start_webhook, start_polling

from FaceGAN import FaceGAN
from EasyStyle import StyleTransfer

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
    dp = Dispatcher(bot, storage=MemoryStorage())
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
    dp = Dispatcher(bot, storage=MemoryStorage())
    dp.middleware.setup(LoggingMiddleware())

logging.basicConfig(level=logging.INFO)


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.answer('Привет, я генерирую лица, у меня есть следующие команды:\n'
                         '/start, /help - вызвать это меню.\n'
                         '/generate - сгенерировать случайное лицо.\n'
                         '/style - применить стиль одной фотографии к другой.\n')


@dp.message_handler(commands=['generate'])
async def generate(message: types.Message):
    generator = FaceGAN()
    logging.debug('Генерирую...')
    await generator.get_image()
    if os.path.isfile(f'faces/fake.jpg'):
        await bot.send_photo(chat_id=message.from_user.id, photo=open('faces/fake.jpg', 'rb'))
    else:
        await message.answer("Упс.. Ошибочка вышла.")


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


class Form(StatesGroup):
    waiting_content = State()
    waiting_style = State()
    magic = State()


@dp.message_handler(commands=['style'])
async def send_welcome(message: types.Message):
    await Form.waiting_content.set()
    await message.answer('Пожалуйста, пришлите фотографию для которой будем менять стиль.\n'
                         'Для отмены используйте /cancel')


@dp.message_handler(state=Form.waiting_content, content_types=['photo'])
async def process_content(message: types.Message, state: FSMContext):
    if not os.path.exists(f'content'):
        os.makedirs('content')
    await message.photo[-1].download(f'content/cnt{str(message.from_user.id)}.jpg')
    await Form.waiting_style.set()
    await message.answer('Пожалуйста, пришлите фотографию c желаемым стилем.\n'
                         'Для отмены используйте /cancel')


@dp.message_handler(state=Form.waiting_style, content_types=['photo'])
async def process_style(message: types.Message, state: FSMContext):
    if not os.path.exists(f'style'):
        os.makedirs('style')
    await message.photo[-1].download(f'style/stl{str(message.from_user.id)}.jpg')
    await process_magic(message, state)


async def process_magic(message: types.Message, state: FSMContext):
    if not os.path.exists(f'transferred'):
        os.makedirs('transferred')
    content_path = f'content/cnt{str(message.from_user.id)}.jpg'
    style_path = f'style/stl{str(message.from_user.id)}.jpg'
    trans_path = f'transferred/image{str(message.from_user.id)}.jpg'

    await message.answer("Я начал работать, подождите около 5 минут.")

    t = threading.Thread(target=lambda msg, content_ph, style_ph, trans_ph
                         : asyncio.run(process_transfer(msg, content_ph, style_ph, trans_ph)),
                         args=(message, content_path, style_path, trans_path))
    t.start()
    await state.finish()


async def process_transfer(message: types.Message, content_path, style_path, trans_path):
    model = StyleTransfer(content_path, style_path, message.from_user.id)
    await model.transfer()

    boto = Bot(token=API_TOKEN)

    if os.path.isfile(trans_path):
        await boto.send_photo(chat_id=message.from_user.id,
                             photo=open(trans_path, 'rb'))
    else:
        await message.answer("Упс.. Ошибочка вышла.")
    await model.clear()
    await boto.close()


@dp.message_handler(state='*', commands='cancel')
@dp.message_handler(Text(equals='cancel', ignore_case=True), state='*')
async def cancel_handler(message: types.Message, state: FSMContext):
    """
    Allow user to cancel any action
    """
    current_state = await state.get_state()
    if current_state is None:
        return

    logging.info('Cancelling state %r', current_state)
    # Cancel state and inform user about it
    await state.finish()
    # And remove keyboard (just in case)
    await message.reply('Отменено.', reply_markup=types.ReplyKeyboardRemove())


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
