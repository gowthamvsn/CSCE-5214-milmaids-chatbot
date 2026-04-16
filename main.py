import os
import importlib


# Securely loads API key in .env file
from dotenv import load_dotenv
load_dotenv()


def list_bots():
    bot_files = os.listdir("bots")
    bots = [os.path.splitext(bot)[0] for bot in bot_files if bot.endswith('.py')]
    return bots


def choose_bot():
    bots = list_bots()
    print("Available bots:")
    for index, bot in enumerate(bots, 1):
        print(f"{index}. {bot}")

    choice = int(input("Choose a bot by entering its number: ")) - 1
    return bots[choice]


def select_bot():
    chosen_bot = choose_bot()
    bot_module = importlib.import_module(f"bots.{chosen_bot}")
    bot_module.main()


if __name__ == "__main__":
    select_bot()
