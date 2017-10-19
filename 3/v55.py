import scipy.stats as sps
import numpy as np
from math import pi, sqrt


class WinerProcess:
	def __init__(self, precision=10000):
		self.__private_precision = precision
		self.__private_xi = np.array([sps.norm.rvs(size=precision+1)])
		self.__private_k = np.arange(1, precision+1)

	def __getitem__(self, times):
    
		times=np.append([], times)
		leftboard = int(max(times)/pi)
		if(leftboard >= len(self.__private_xi)):
			self.__private_xi = np.append(self.__private_xi, sps.norm.rvs(size=(self.__private_precision+1)*(leftboard-len(self.__private_xi)+1))).reshape(leftboard+1, self.__private_precision+1)
		return [self._X_t(i) for i in times]
    
	def _X_t(self, t):
		
		leftboard = int(t/pi)
		realt = t - pi*leftboard 
        
		answer = np.dot(self.__private_xi[:leftboard, 0], np.arange(1, leftboard+1)) * sqrt(pi)
		
		answer += self.__private_xi[leftboard][0]*realt/sqrt(pi)
		
		answer += sqrt(2/pi) * np.dot(np.sin(self.__private_k * t)/self.__private_k, self.__private_xi[leftboard][1:])
		return answer


def winer_proccess_path(end_time, step, precision=10000):
    times = np.arange(0, end_time, step)
    values = np.zeros_like(times, dtype=float)
    
    mustbeadded = 0.
    
    leftboard = -1
    
    xi = [0]
    i = 0
    
    k = np.arange(1, precision+1)
    
    for t in times:
        
        board = int(t/pi)
        
        if(board > leftboard):
            
            xi_0 = sps.norm.rvs(size = board-leftboard-1)
            
            mustbeadded += np.dot(xi_0, np.arange(leftboard+1, board))*sqrt(pi)
            
            mustbeadded += leftboard*xi[0]*sqrt(pi)
            
            xi = sps.norm.rvs(size = precision+1)
            
        
        values[i] = mustbeadded + xi[0]*t/sqrt(pi) + sqrt(2/pi)*np.dot(np.sin(k*t)/k, xi[1:])
        i +=1
    return times, values
