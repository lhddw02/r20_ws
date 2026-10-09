"""Bounded serial I/O shared by the ROS bridge and zero-motion probe."""

import os

import serial


def open_port(port, baud):
    options = dict(
        port=port, baudrate=baud, bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
        timeout=0, write_timeout=0.1,
        xonxoff=False, rtscts=False, dsrdtr=False,
    )
    if os.name == 'posix':
        options['exclusive'] = True
    return serial.Serial(**options)


def write_packet(port, packet):
    if port.write(packet) != len(packet):
        raise serial.SerialTimeoutException('Incomplete serial packet write')
