import sys

def convert_shader_to_c(input_file, output_file, chunk_size=12*1024):
    with open(input_file, 'r') as shader:
        shader_content = shader.read()

    chunks = [shader_content[i:i+chunk_size] for i in range(0, len(shader_content), chunk_size)]
    shadername = input_file  # input_file.split('/')[-1]
    shadername = shadername.replace('-', '_').replace('.', '_').replace(' ', '_').replace('/', '_')

    with open(output_file, 'w') as output:
        output.write('// Generated C source file containing shader\n\n')
        output.write('#include <string>\n\n')
        i = 0
        for chunk in chunks:
            output.write(f'const std::string {shadername}_chunk{i} = R"(\n')
            output.write(f'{chunk}')
            output.write(')";\n\n')

        output.write(f'const std::string {shadername}_src = \n')
        for i in range(len(chunks)):
            output.write(f'    {shadername}_chunk{i}')
            if i < len(chunks) - 1:
                output.write(' +\n')
        output.write(';\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_shader_file> <output_c_file>")
        sys.exit(1)

    input_shader = sys.argv[1]
    output_c = sys.argv[2]
    convert_shader_to_c(input_shader, output_c)
