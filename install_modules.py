# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 11:37:24 2018

@author: dominic
"""
from subprocess import check_output
import platform
import json

from distutils.util import convert_path 

sys = platform.system()
newLineChar = ''
if(sys == 'Linux' or sys == 'Darwin'):
    newLineChar = '/'
elif(sys == 'Windows'):
    newLineChar = '&'

modules = []
moduleVersion = {}
moduleDependencies = {}
alreadyInstalled = {}

with open("modules.json","r") as config_file:
    modules=json.load(config_file)

for module in modules:
    alreadyInstalled[module] = False
    main_ns = {}
    ver_path = convert_path(module+'/'+module+'/version.py')
    with open(ver_path) as ver_file:
        exec(ver_file.read(),main_ns)
        moduleVersion[module] = main_ns['__version__']
       
with open("moduleDependencies.json","r") as config_file:
    moduleDependencies = json.load(config_file)

def cleanDistributions():
	check_output("rm -rf /dist/")
    
def wheelandinstall(m):
    if(not alreadyInstalled[m]):
        for dependency in moduleDependencies[m]:
            if(not alreadyInstalled[dependency]):
                wheelandinstall(dependency)
        check_output("cd "+m+" "+newLineChar+" python setup.py clean "+newLineChar+" python setup.py clean --all bdist_wheel -d ../dist/ clean --all",shell=True)
        check_output("pip install dist/"+m+"-"+moduleVersion[m]+"-py3-none-any.whl")
        alreadyInstalled[m] = True	
        
for module in modules:
	wheelandinstall(module)
	print("Wheeled and installed "+module+".")