import pygame
pygame.init()

# Map style names to actual font file names (handles camelCase like ExtraLight)
style_to_filename = {
    "regular": "Regular",
    "bold": "Bold",
    "thin": "Thin",
    "extralight": "ExtraLight",
}
sizes = [i for i in range(100)]

fonts = {
    f"{style}{size}": pygame.font.Font(f"fonts/Montserrat-{style_to_filename[style]}.ttf", size)
    for style in style_to_filename for size in sizes
}
