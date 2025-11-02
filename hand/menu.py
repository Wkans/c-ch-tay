import subprocess
import sys
import asyncio
import platform

async def main():
    while True:
        print("\nMenu:")
        print("1. Run Virtual Mouse")
        print("2. Run Predict Live")
        print("0. Exit")
        choice = input("Enter your choice: ")

        try:
            if choice == "1":
                print("Running Virtual Mouse...")
                subprocess.run([sys.executable, "virtualmouse.py"], check=True)
            elif choice == "2":
                print("Running Predict Live...")
                subprocess.run([sys.executable, "predict_live.py"], check=True)
            elif choice == "0":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")
        except subprocess.CalledProcessError as e:
            print(f"Error running script: {e}")
        except FileNotFoundError:
            print("Error: The specified script file was not found. Ensure 'virtualmouse.py' or 'predictlive.py' is in the same directory.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())