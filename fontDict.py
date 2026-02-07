import pygame
pygame.init()

styles = ["regular", "bold", "thin", "extralight"]
sizes = [i for i in range(100)]

# fonts = {
#     f"{style}{size}": pygame.font.Font(f"fonts/Montserrat-{style.capitalize()}.ttf", size) for style in styles for size in sizes}
# 替换原来的 fonts = {...} 这一行（单行字典推导）
STYLE_TO_FILE = {
    "regular": "Regular",
    "bold": "Bold",
    "thin": "Thin",
    "extralight": "ExtraLight",  # 注意 L 要大写
}

fonts = {
    f"{style}{size}": pygame.font.Font(
        f"fonts/Montserrat-{STYLE_TO_FILE[style]}.ttf", size
    )
    for style in styles
    for size in sizes
}