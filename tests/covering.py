from shapely import Point, Polygon, LineString

class ball:
    '''
        Is the object that defines the mesoscale of the model. It defines the radius over which letting $\omega$ big enough, we have that $\lambda$
        is small enough so that vt<r. If I do not have this condition, I have to change the radius of the ball.
    '''
    def __init__(self,center,radius):
        self.center = center
        self.radius = radius
        thetha = np.linspace(0,2*np.pi,100)
        self.circle = Polygon([(center.x+radius*np.cos(t),center.y+radius*np.sin(t)) for t in thetha])
        self.area = self.circle.area
        self.neighbors = []

    def add_node_to_neighborhood(self,point):
        '''
            Input:
                point = np.array([x,y])
            Output:
                Add point to neighborhood
        '''

        if self.circle.contains(Point(point)):
            self.neighbors.append(point)
        else:
            pass


    