from time import time
import sys


def read_write_lines_hashes(file_name, out_file_name, max_length=10):
    """
    For use with password files from Hashes.org. This function pads passwords with the '|' character to a specified length and removes the passwords greater than the specified length.
    """
    passwords = []
    with open(file_name, 'r', errors='ignore') as f:
        for line in f:
            password = line.strip().split(':')[1]
            if len(password) <= max_length:
                passwords.append(password.ljust(max_length, '|'))

    with open(out_file_name, "w") as f:
        for password in passwords:
            f.write(password + "\n")


def read_write_lines_other(file_name: str, out_file_name: str, max_length=10):
    """
    Pads passwords with the '|' character to a specified length. Removes the passwords greater than the specified length.
    """
    passwords = []
    with open(file_name, 'r', errors='ignore') as f:
        for line in f:
            password = line.strip()
            if len(password) <= max_length:
                passwords.append(password.ljust(max_length, '|'))

    with open(out_file_name, "w") as f:
        for password in passwords:
            f.write(password + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[3] not in ["hashes", "other"]:
        print("usage: python3 data.py <input_file> <output_file> hashes")

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    source = sys.argv[3]

    t = time()
    if source == "hashes":
        read_write_lines_hashes(input_file, output_file, max_length=10)
    else:
        read_write_lines_other(input_file, output_file, max_length=10)
    print(f"{time() - t:.2f} seconds")


# gcloud compute scp linkedin_processed.txt nvijayakumar@gpu-instance:~/gcp-gan/Data
# gcloud compute scp rockyou_processed.txt nvijayakumar@gpu-instance:~/gcp-gan/Data
# gcloud compute scp dubsmash_processed.txt nvijayakumar@gpu-instance:~/gcp-gan/Data