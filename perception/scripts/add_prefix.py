import os
import argparse

def add_prefix_to_files(folder_path: str, prefix: str):
    # Переконуємося, що шлях існує
    if not os.path.exists(folder_path):
        print(f"Помилка: Папка '{folder_path}' не існує!")
        return

    # Забезпечуємо роздільник (наприклад, 'v1_')
    clean_prefix = prefix.strip()
    if not clean_prefix.endswith("_"):
        clean_prefix += "_"

    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    if not files:
        print(f"У папці '{folder_path}' немає файлів для перейменування.")
        return

    renamed_count = 0
    skipped_count = 0

    print(f"Починаємо перейменування у папці: {folder_path}")
    print(f"Префікс: '{clean_prefix}'\n")

    for filename in files:
        # Уникаємо повторного додавання префікса
        if filename.startswith(clean_prefix):
            skipped_count += 1
            continue

        old_path = os.path.join(folder_path, filename)
        new_filename = f"{clean_prefix}{filename}"
        new_path = os.path.join(folder_path, new_filename)

        os.rename(old_path, new_path)
        renamed_count += 1

    print(f"Готово! Перейменовано: {renamed_count} файлів. Пропущено (вже мали префікс): {skipped_count}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Додавання префікса (v1_, v2_ тощо) до всіх файлів у папці.")
    parser.add_argument("--folder", "-f", type=str, help="Шлях до цільової папки з файлами")
    parser.add_argument("--prefix", "-p", type=str, help="Префікс (наприклад: v1, v2)")

    args = parser.parse_args()

    # Якщо аргументи не вказані у консолі, запитуємо інтерактивно
    folder = args.folder or input("Введіть шлях до папки (наприклад, images/train/video1): ").strip()
    prefix = args.prefix or input("Введіть префікс (наприклад, v1): ").strip()

    add_prefix_to_files(folder, prefix)