# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 11:37:24 2018

@author: dominic
"""
from subprocess import check_output
import platform
import json

sys = platform.system()
newLineChar = ''
if(sys == 'Linux' or sys == 'Darwin'):
    newLineChar = '/'
elif(sys == 'Windows'):
    newLineChar = '&'

modules = []
with open("modules.json","r") as config_file:
    modules=json.load(config_file)

alreadyInstalled = {}
for module in modules:
    alreadyInstalled[module] = False
    
def test(m):
    print("")
    print("Beginning tests for "+m+".")
    check_output("cd "+m+" "+newLineChar+" python setup.py test",shell=True)
    print("Finished tests for "+m+".")
    print("")
for module in modules:
	test(module)