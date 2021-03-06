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

from models.FaceGAN_dir.FaceGAN import FaceGAN
from models.EasyStyle_dir.EasyStyle import StyleTransfer
from models.CycleGAN_dir.CycleGAN import Summer2Winter

# Easier to test it with pooling
MODE = str(os.getenv('HEROKU'))

# configuration
if MODE == 'DEPL':
    # webhook
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
    dp = Dispatcher(bot, storage=MemoryStorage())
    dp.middleware.setup(LoggingMiddleware())
else:
    # polling
    API_TOKEN = str(os.getenv('BOT_TOKEN'))
    HEROKU_APP_NAME = str(os.getenv('HEROKU_APP_NAME'))

    bot = Bot(token=API_TOKEN)
    dp = Dispatcher(bot, storage=MemoryStorage())
    dp.middleware.setup(LoggingMiddleware())

logging.basicConfig(level=logging.INFO)


async def on_startup(dp):
    logging.warning('Starting connection.')
    if MODE == 'DEPL':
        await bot.delete_webhook()
        await bot.set_webhook(WEBHOOK_URL, drop_pending_updates=True)


async def on_shutdown(dp):
    logging.warning('Shutting down..')
    # Close DB connection (if used)
    await dp.storage.close()
    await dp.storage.wait_closed()

    logging.warning('Bye!')


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.answer('Привет, у меня есть следующие команды:\n'
                         '/start, /help - вызвать это меню.\n'
                         '/generate - сгенерировать случайное лицо.\n'
                         '/style - применить стиль одной фотографии к другой. Это может занять до 35 минут.\n'
                         '/sum2win - превратить лето в зиму.\n')


@dp.message_handler(commands=['generate'])
async def generate(message: types.Message):
    # generate random face
    generator = FaceGAN()
    logging.debug('Генерирую...')
    await generator.get_image()
    if os.path.isfile(f'models/FaceGAN_dir/faces/fake.jpg'):
        await bot.send_photo(chat_id=message.from_user.id, photo=open('models/FaceGAN_dir/faces/fake.jpg', 'rb'))
    else:
        await message.answer("Упс.. Ошибочка вышла.")


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


# Easy_style_transfer start

# start -> waiting_content -(got a picture)-> waiting_style -(got a picture)-> (running) -> finish (gives the answer)
#                          \                       \
#                           -> finish (canceled)    -> finish (canceled)
class FormEasyStyle(StatesGroup):
    waiting_content = State()
    waiting_style = State()


@dp.message_handler(commands=['style'])
async def send_welcome(message: types.Message):
    await FormEasyStyle.waiting_content.set()
    await message.answer('Пожалуйста, пришлите фотографию для которой будем менять стиль.\n'
                         'Это может занять до 35 минут.\n'
                         'Для отмены используйте /cancel')


@dp.message_handler(state=FormEasyStyle.waiting_content, content_types=['photo'])
async def process_content(message: types.Message):
    if not os.path.exists(f'models/EasyStyle_dir/content'):
        os.makedirs('models/EasyStyle_dir/content')
    await message.photo[-1].download(f'models/EasyStyle_dir/content/cnt{str(message.from_user.id)}.jpg')
    await FormEasyStyle.waiting_style.set()
    await message.answer('Пожалуйста, пришлите фотографию c желаемым стилем.\n'
                         'Это может занять до 35 минут.\n'
                         'Для отмены используйте /cancel')


@dp.message_handler(state=FormEasyStyle.waiting_style, content_types=['photo'])
async def process_style(message: types.Message, state: FSMContext):
    if not os.path.exists(f'models/EasyStyle_dir/style'):
        os.makedirs('models/EasyStyle_dir/style')
    await message.photo[-1].download(f'models/EasyStyle_dir/style/stl{str(message.from_user.id)}.jpg')
    # runs the model and sends an answer
    await process_magic(message, state)


async def process_magic(message: types.Message, state: FSMContext):
    if not os.path.exists(f'models/EasyStyle_dir/transferred'):
        os.makedirs('models/EasyStyle_dir/transferred')
    content_path = f'models/EasyStyle_dir/content/cnt{str(message.from_user.id)}.jpg'
    style_path = f'models/EasyStyle_dir/style/stl{str(message.from_user.id)}.jpg'
    trans_path = f'models/EasyStyle_dir/transferred/image{str(message.from_user.id)}.jpg'

    await message.answer("Я начал работать, подождите около 30 минут.")

    # avoiding timeouts
    t = threading.Thread(target=lambda msg, content_ph, style_ph, trans_ph
                         : asyncio.run(process_transfer(msg, content_ph, style_ph, trans_ph)),
                         args=(message, content_path, style_path, trans_path))
    t.start()
    await state.finish()


async def process_transfer(message: types.Message, content_path, style_path, trans_path):
    model = StyleTransfer(content_path, style_path, trans_path, message.from_user.id)
    await model.transfer()

    boto = Bot(token=API_TOKEN)

    if os.path.isfile(trans_path):
        await boto.send_photo(chat_id=message.from_user.id,
                             photo=open(trans_path, 'rb'))
    else:
        await message.answer("Упс.. Ошибочка вышла.")
    await model.clear()
    await boto.close()


# Easy_style_transfer end
# CycleGAN_dir transfer start

# start -> waiting_content -(got a picture)-> (running) -> finish (gives the answer)
#                          \
#                           -> finish (canceled)
class FormSum2Win(StatesGroup):
    waiting_content = State()


@dp.message_handler(commands=['sum2win'])
async def send_welcome(message: types.Message):
    await FormSum2Win.waiting_content.set()
    await message.answer('Пожалуйста, пришлите фотографию летнего пейзажа.\n'
                         'Для отмены используйте /cancel')


@dp.message_handler(state=FormSum2Win.waiting_content, content_types=['photo'])
async def process_content(message: types.Message, state: FSMContext):
    if not os.path.exists(f'models/CycleGAN_dir/real'):
        os.makedirs('models/CycleGAN_dir/real')
    await message.photo[-1].download(f'models/CycleGAN_dir/real/cnt{str(message.from_user.id)}.jpg')
    # runs the model and sends an answer
    await process_make_it_winter(message, state)


async def process_make_it_winter(message: types.Message, state: FSMContext):
    if not os.path.exists(f'models/CycleGAN_dir/fake'):
        os.makedirs('models/CycleGAN_dir/fake')
    content_path = f'models/CycleGAN_dir/real/cnt{str(message.from_user.id)}.jpg'
    trans_path = f'models/CycleGAN_dir/fake/image{str(message.from_user.id)}.jpg'

    await message.answer("Я начал работать, подождите пару минут.")

    # avoiding timeouts
    t = threading.Thread(target=lambda msg, content_ph, trans_ph
                         : asyncio.run(process_winter_transfer(msg, content_ph, trans_ph)),
                         args=(message, content_path, trans_path))
    t.start()
    await state.finish()


async def process_winter_transfer(message: types.Message, content_path, trans_path):
    model = Summer2Winter(content_path, trans_path)
    await model.get_image()

    boto = Bot(token=API_TOKEN)

    if os.path.isfile(trans_path):
        await boto.send_photo(chat_id=message.from_user.id,
                             photo=open(trans_path, 'rb'))
    else:
        await message.answer("Упс.. Ошибочка вышла.")
    await model.clear()
    await boto.close()


def start():
    if MODE == 'DEPL':
        # I use a webhook at heroku
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
        # and a polling locally
        start_polling(
            dispatcher=dp,
            skip_updates=True
        )


if __name__ == '__main__':
    start()
