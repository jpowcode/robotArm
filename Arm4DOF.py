import math
import numpy as np
import scipy.optimize  
import timeit

class Arm3Link:
    
    def __init__(self, q=None, q0=None, L=None):
        """Set up the basic parameters of the arm.
        All lists are in order [shoulder, elbow, wrist].
        
        :param list q: the initial joint angles of the arm
        :param list q0: the default (resting state) joint configuration
        :param list L: the arm segment lengths
        """
        # initial joint angles
        if q is None: q = [math.pi/4, math.pi/4, 0, 0]
        self.q = q
        # some default arm positions
        if q0 is None: q0 = np.array([math.pi/4, math.pi/4, 0, 0]) 
        self.q0 = q0
        # arm segment lengths
        if L is None: L = np.array([1, 1, 1, 1]) 
        self.L = L
        
        self.max_angles = [math.pi, math.pi, math.pi, math.pi ]
        self.min_angles = [0, 0, -math.pi, -math.pi]

    def get_xy(self, q=None):
        """Returns the corresponding hand xy coordinates for 
        a given set of joint angle values [shoulder, elbow, wrist], 
        and the above defined arm segment lengths, L
        
        :param list q: the list of current joint angles
        :returns list: the [x,y] position of the arm
        """
        if q is None: q = self.q

        x = self.L[0]*np.cos(q[0]) + \
            self.L[1]*np.cos(q[0]+q[1]) + \
            self.L[2]*np.cos(q[0]+q[1]+q[2]) + \
            self.L[3]*np.cos(np.sum(q)) 
            

        y = self.L[0]*np.sin(q[0]) + \
            self.L[1]*np.sin(q[0]+q[1]) + \
			self.L[2]*np.sin(q[0]+q[1]+q[2]) + \
            self.L[3]*np.sin(np.sum(q))

        return [x, y]

    def inv_kin(self, xy):
        """This is just a quick write up to find the inverse kinematics
        for a 3-link arm, using the SciPy optimize package minimization function.

        Given an (x,y) position of the hand, return a set of joint angles (q)
        using constraint based minimization, constraint is to match hand (x,y), 
        minimize the distance of each joint from it's default position (q0).
        
        :param list xy: a tuple of the desired xy position of the arm
        :returns list: the optimal [shoulder, elbow, wrist] angle configuration
        """

        def distance_to_default(q, *args): 
            """Objective function to minimize
            Calculates the euclidean distance through joint space to the default
            arm configuration. The weight list allows the penalty of each joint 
            being away from the resting position to be scaled differently, such
            that the arm tries to stay closer to resting state more for higher 
            weighted joints than those with a lower weight.
            
            :param list q: the list of current joint angles
            :returns scalar: euclidean distance to the default arm position
            """
            # weights found with trial and error, get some wrist bend, but not much
            weight = [1, 1, 1.3, 1] 
            return np.sqrt(np.sum([(qi - q0i)**2 * wi
                for qi,q0i,wi in zip(q, self.q0, weight)]))

        def x_constraint(q, xy):
            """Returns the corresponding hand xy coordinates for 
            a given set of joint angle values [shoulder, elbow, wrist], 
            and the above defined arm segment lengths, L
            
            :param list q: the list of current joint angles
            :returns: the difference between current and desired x position
            """
            x = ( self.L[0]*np.cos(q[0]) + self.L[1]*np.cos(q[0]+q[1]) + 
                self.L[2]*np.cos(q[0]+q[1]+q[2]) + self.L[3]*np.cos(np.sum(q)) ) - xy[0]
            return x

        def y_constraint(q, xy): 
            """Returns the corresponding hand xy coordinates for 
            a given set of joint angle values [shoulder, elbow, wrist], 
            and the above defined arm segment lengths, L
            
            :param list q: the list of current joint angles
            :returns: the difference between current and desired y position
            """
            y = ( self.L[0]*np.sin(q[0]) + self.L[1]*np.sin(q[0]+q[1]) + 
                self.L[2]*np.sin(q[0]+q[1]+q[2]) + self.L[3]*np.sin(np.sum(q)) ) - xy[1]
            return y

        return scipy.optimize.fmin_slsqp( func=distance_to_default, 
            x0=self.q, eqcons=[x_constraint, y_constraint], 
            args=(xy,), iprint=0) # iprint=0 suppresses output


def findJointPos(x, y):
   
    arm = Arm3Link()
    xy = [x, y]
    # run the inv_kin function, get the optimal joint angles
    return arm.inv_kin(xy)
    

