import socket
import pickle
import asyncio
import struct
import os
from time import sleep
from config import *

def connect_send_socket(dst_ip, dst_port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    while s.connect_ex((dst_ip, dst_port)) != 0:
        sleep(0.5)

    return s

def connect_get_socket(listen_ip, listen_port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    while True:
        try:
            s.bind((listen_ip, listen_port))
            break
        except OSError as e:
            print(e)
            print("**OSError**", listen_ip, listen_port)
            sleep(0.7)
    s.listen(1)

    conn, _ = s.accept()

    return conn

def send_data_socket(data, s):
    data = pickle.dumps(data)
    s.sendall(struct.pack(">I", len(data)))
    s.sendall(data)

def get_data_socket(conn):
    data_len = struct.unpack(">I", conn.recv(4))[0]
    data = conn.recv(data_len, socket.MSG_WAITALL)
    # data = recv_basic(conn)
    recv_data = pickle.loads(data)

    return recv_data

def kill_port(port):
    command = '''kill -9 $(netstat -nlp | grep :''' + str(port) + ''' | awk '{print $7}' | awk -F"/" '{ print $1 }')'''
    os.system(command)

async def send_data(config, dst_ip, dst_port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        while s.connect_ex((dst_ip, dst_port)) != 0:
            sleep(1)
        data = pickle.dumps(config, protocol=pickle.HIGHEST_PROTOCOL)
        s.send(data)


async def get_data(listen_port, listen_ip=socket.gethostbyname(socket.gethostname())):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        while True:
            try:
                s.bind((listen_ip, listen_port))
                break
            except OSError:
                print("**OSError**",listen_ip,listen_port)
                kill_port(listen_port)
        s.listen(1)
        while True:
            try:
                conn, _ = s.accept()
                break
            except:
                await asyncio.sleep(0.5)
                continue
        data = recv_basic(conn)
        config = pickle.loads(data)
        return config


def recv_basic(conn):
    total_data = b''
    while True:
        data = conn.recv(20480)
        if not data:
            break
        total_data = total_data + data
    return total_data
