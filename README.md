# style_transfer_bot
Этот телеграм бот написан с использованием aiogram, pytorch и torchvision, он имеет следующие функции:

По командам /start или /help выводится меню со списком всех возможных команд.

По команде /generate бот присылает сгенерированное лицо.

По команде /style бот переносит стиль одной фотографии на другую.

По команде /sum2win бот превращает фотографию летнего пейзажа в зимний.

## Как работает команда generate

При получении запроса запускается генератор из GAN'а последнего домашнего задания, далее сгенерированное изображение сохраняется в директорию "faces", после чего отправляется пользователю.

## Как работает команда style

При получении запроса мы попадаем в первое состояние конечного автомата. Из первого состояния есть два пути. Первый - прислать боту фотографию к которой вы желаете применить какой-то стиль. Второй - использовать команду /cancel, тогда вы перейдете в конечное состояние автомата. Если вы прислали фотографию, то бот перейдет во второе состояние, из которого тоже есть два пути: Первый - прислать боту фотографию с желаемым стилем. Второй - использовать команду /cancel, тогда вы перейдете в конечное состояние автомата. Если вы присылаете фотографию стиля, бот начинает работу над изображением и через какое-то время присылает вам результат.

![alt text](https://github.com/MaloYY/style_transfer_bot/blob/master/images/style_automat_w.png)

В основе модели переноса стиля стоит предложенная на семинаре "медленная" модель. Разница лишь в том, что я заменил VGG19 на VGG16 при этом я загружаю нетренированную модель, а веса храню только для той части модели, которую я использую. Полученные изображения хранятся соответсвенно в директориях "content" и "style", а итоговое изображение сохраняется в директорию "transferred", после полученное изображение отправляется пользователю. В названиях файлов используется id пользователя, чтобы не происходила путаница при обработке и отправки.

## Как работает команда sum2win

При получении запроса мы попадаем в первое состояние конечного автомата. Из первого состояния есть два пути. Первый - прислать боту фотографию летнего пейзажа. Второй - использовать команду /cancel, тогда вы перейдете в конечное состояние автомата. Если вы прислали фотографию, бот пришлет в ответ вашу фотографию в зимнем стиле.

![alt text](https://github.com/MaloYY/style_transfer_bot/blob/master/images/sum2win_automat_w.png)

Тут я использовал CycleGAN в основе лежала реализация и веса человека с ником [junyanz](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/f13aab8148bd5f15b9eb47b690496df8dadbab0c). Но код пришлось сильно переработать, так как у автора создание и вызов модели происходят самописным образом, который совершенно не подходит для использования под капотом телеграм бота. Я переписал создание и вызов модели в более подходящей мне форме. Полученное изображение сохраняется в директорию "real", а сгенерированное в директорию "fake", после чего сгенерированное изображение отправляется пользователю. Опять же в названиях файлов используется id пользователя.

## Дополнение

Сделал через BotFather своему боту описание, аватарку и добавил команды в список команд.

## Скриншоты работы

![alt text](https://github.com/MaloYY/style_transfer_bot/blob/master/images/bot_works_t.png)