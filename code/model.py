import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import sys
import pylab

class SML():
    def __init__(self, cas_path, gra_path, start_node, iters, mode):
        self.cas_path = cas_path
        self.gra_path = gra_path
        self.start_node = start_node
        self.iters = iters
        self.mode = mode
        self.low = 0.01
        self.medium = 0.05
        self.high = 0.2
        self.state = [] # record the state of nodes
        self.gra_dict = {} # u -> v : probability
        self.out = {} # the nodes' out-neighbour 

    def read_cascade_data(self):
        fcas = open(self.cas_path)
        lcas = []
        t = 0
        for line in fcas:
            line = line.split(',')
            if int(line[1]) == t:
                lcas[t-1] += 1
            elif int(line[1]) == t+1:
                lcas.append(1)
                t += 1
        fcas.close()
        return np.array(lcas), np.array([i+1 for i in range(t)]), t

    def read_graph_data(self):
        fgra = open(self.gra_path)
        temp = -1
        temp_list = []
        for line in fgra:
            line = line.split(',')
            if len(line) != 3:
                continue
            if int(line[0]) != temp:
                if temp != -1:
                    self.out[temp] = temp_list
                    temp_list = []
                temp = int(line[0])
            temp_list.append(int(line[1]))
            self.gra_dict[(int(line[0]), int(line[1]))] = float(line[2])
        self.out[temp] = temp_list
        fgra.close()
        return len(self.out)

    def poly(self, x, y):
        z = np.polyfit(x, y, 2)
        fit = z[0] * np.power(x,2) + z[1] * x + z[2]
        return z, fit

    def dfit(self, y):
        z, fit = self.poly(self.x, y)
        y_err = y - self.lcas
        mean_x = np.mean(self.x)
        n = len(self.x)
        t = 1.0528 # 68.27% - one standard deviation, self.T = 12
        s_err = np.sum(np.power(y_err, 2))
        confs = t*np.sqrt((s_err/(n-2))*(1.0/n+(np.power((self.x-mean_x),2)/((np.sum(np.power(self.x,2)))-n*(np.power(mean_x,2))))))
        lower = fit - abs(confs)
        upper = fit + abs(confs)
        return lower, upper, fit
    def simulate_one(self):
        """
        state: 0 - inactive
        state: 1 - active
        state: 2 - previous active
        function:  once simulation
        """
        self.active = set()
        self.active.add(self.start_node)
        self.be_active = set()
        for act in self.active:
            self.state[act] = 1
        for t in range(1, self.T+1):
            bef_node = set()
            for act in self.active:
                bef_node.update(self.out[act])
                self.state[act] = 2
            counts = 0
            for node in bef_node:
                #print node
                p_inac = 1.0
                if self.state[node] == 1 or self.state[node] == 2:
                    continue
                for act in self.active:
                    if self.gra_dict.has_key((act,node)):
                        if self.mode == 0:
                            p_inac *= 1.0 - self.gra_dict[(act, node)]
                        elif self.mode == 1:
                            p_inac *= 1.0 - self.low
                        elif self.mode == 2:
                            p_inac *= 1.0 - self.medium
                        else:
                            p_inac *= 1.0 - self.high
                        #p_inac *= 1.0 - 0.1
                p_inac = 1.0 - p_inac
                if p_inac > np.random.rand():
                    self.state[node] = 1
                    counts += 1
                    self.be_active.add(node)
            self.active = set(self.be_active)
            self.be_active = set()
            self.evolution[t-1] += counts
        #return evolution
    def simulate(self):
        self.lcas, self.x, self.T = self.read_cascade_data()
        _, fit_c = self.poly(self.x, self.lcas)
        self.lens = self.read_graph_data()
        fig, ax = plt.subplots(1,1)
        ax.set_xlim(0,self.T+1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.7, box.height])
        fig.show()
        fig.canvas.draw()
        self.evolution = np.zeros(self.T)
        md = ''
        if self.mode == 0:
            md = 'heterogeneous'
        elif self.mode == 1:
            md = 'low'
        elif self.mode == 2:
            md = 'medium'
        else:
            md = 'high'
        for it in range(self.iters):
            #print 'it: ' + str(it)
            self.state = [0] * self.lens
            self.simulate_one()
            if it %10 == 0:
                print self.evolution/(it+1)
            lower, upper, fit = self.dfit(self.evolution/(it+1))
            ax.clear()
            ax.set_xlabel('discrete time ')
            ax.set_ylabel('number of activation')
            ax.set_xlim(0, self.T+1)
            
            ax.plot(self.x, self.lcas, 'bo', label='cascade data point')
            ax.plot(self.x, fit_c, 'b-', label='cascade regression line')
            ax.plot(self.x, self.evolution/(it+1), 'ro',label='simulation data point')
            ax.plot(self.x, fit, 'r-', label='simulation regression line')
            ax.plot(self.x, lower, 'r--', label='lower confidence limit (68.27%)')
            ax.plot(self.x, upper, 'r--', label='upper confidence limit (68.27%)')
            ax.plot([0],[0],'w', label='mode = '+md)
            ax.plot([0],[0],'w', label='iter = '+str(it+1))
            legend = ax.legend(loc='center left', shadow=False, fontsize='x-small', bbox_to_anchor=(1,0.5))
            legend.get_frame().set_facecolor('w')
            fig.canvas.draw()
        pylab.savefig('simulate_'+md+'.png')
        #plt.pause(10)
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'you need input a parameter'
        print '0 - Heterogeneous'
        print '1 - Low '
        print '2 - Medium'
        print '3 - High'
        exit()
    sml = SML('cascade.txt', 'graph.txt', 448, 1000, int(sys.argv[1]))
    sml.simulate()

