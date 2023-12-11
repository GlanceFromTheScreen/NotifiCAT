from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


def get_main_kb() -> ReplyKeyboardMarkup:
    """
    фабрика клавиатуры главного меню
    :return:
    """
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(KeyboardButton('Add notification'), KeyboardButton('Edit future plans')) \
        .add(KeyboardButton('List of plans'), KeyboardButton('List of completed plans'))
    return kb


def get_what_to_change_kb() -> ReplyKeyboardMarkup:
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(KeyboardButton('Description')) \
        .add(KeyboardButton('Date'), KeyboardButton('Time')) \
        .add(KeyboardButton('Mark as \'DONE\''), KeyboardButton('Edit periodicity')) \
        .add(KeyboardButton('Delete NotifiCATion'), KeyboardButton('Back to main menu'))

    return kb


def get_files_update_kb() -> ReplyKeyboardMarkup:
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(KeyboardButton('Добавить новый'), KeyboardButton('Удалить имеющийся')) \
        .add(KeyboardButton('Вернуться в главное меню'))

    return kb


def get_done_tasks_kb() -> ReplyKeyboardMarkup:
    """
    фабрика клавиатуры главного меню
    :return:
    """
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(KeyboardButton('Return plan to \'UNDONE\''), KeyboardButton('Back to main menu'))

    return kb


def get_back_kb() -> ReplyKeyboardMarkup:
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(KeyboardButton('Back to main menu'))

    return kb


def get_ikb_with_notifications(list_of_notifications: list) -> InlineKeyboardMarkup:
    ikb = InlineKeyboardMarkup(row_width=2)
    for i in range(len(list_of_notifications)):
        noty = list_of_notifications[i][1] + list_of_notifications[i][2] + list_of_notifications[i][3]
        ikb.add(InlineKeyboardButton(text=f'{noty}',
                                     callback_data=f'{list_of_notifications[i][0]}'))
    return ikb


def get_ikb_with_filenames(list_of_files: list) -> InlineKeyboardMarkup:
    ikb = InlineKeyboardMarkup(row_width=2)
    for i in range(len(list_of_files)):
        ikb.add(InlineKeyboardButton(text=f'{list_of_files[i]}',
                                     callback_data=f'{list_of_files[i]}'))
    return ikb


def get_ikb_with_confirmation() -> InlineKeyboardMarkup:
    ikb = InlineKeyboardMarkup(row_width=1)
    ikb.add(InlineKeyboardButton(text=f'Create notification',
                                 callback_data=f'create_confirm'))\
       .add(InlineKeyboardButton(text=f'Cancel notification',
                                 callback_data=f'cancel_confirm'))
    return ikb
