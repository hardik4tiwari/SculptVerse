import argparse
import trimesh

def convert_ply_to_obj(input_file, output_file):
    # Load the PLY file
    mesh = trimesh.load(input_file)
    
    # Export to OBJ file
    mesh.export(output_file)
    print(f"Converted {input_file} to {output_file}")

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Convert PLY file to OBJ format.')
    parser.add_argument('input', type=str, help='Path to the input PLY file.')
    parser.add_argument('output', type=str, help='Path to the output OBJ file.')
    
    args = parser.parse_args()
    
    # Convert the file
    convert_ply_to_obj(args.input, args.output)

if __name__ == '__main__':
    main()
