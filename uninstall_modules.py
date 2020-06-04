# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 11:37:24 2018

@author: dominic
"""
import json
from subprocess import PIPE, Popen, STDOUT

modules = []
with open("modules.json","r") as config_file:
    modules=json.load(config_file)
for m in modules:
    p = Popen(["pip","uninstall",m],stdin=PIPE, stdout=PIPE, stderr = STDOUT,encoding = 'utf-8')
    p.communicate('y')[0].rstrip()
    print("Module "+m+" uninstalled.")