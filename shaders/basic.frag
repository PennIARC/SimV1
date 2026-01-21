#version 330 core

out vec4 f_color;
in vec2 v_text;

uniform sampler2D Texture;

void main() {
    f_color = texture(Texture, v_text);
}