import os, sys
if 'SUMO_HOME' not in os.environ:
     SUMO_HOME=os.path.join(os.sep, 'home','sumo')
     sys.path.append('SUMO_HOME')	

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)
 


