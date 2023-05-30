# Metric Error codes

Some metrics cannot be computed in every possible situation, therefore, they have error codes (negative values) to 
show why it was not possible to calculate the chosen metrics.

## Gap time
* -1: No intersection found
* -2: intersection for some reasons not valid (the entering or leaving times have issues. this happens for example if a 
car following scenario happens: both vehicles have entered the intersection but didn't leave) 
* -3: intersecting angle is too small for being an intersection scenario
* -4: Vehicles are too far apart (more than 25 m which is braking distance for 50 km/h)
* -5: one vehicle is standing still -> then there is no intersection to predict
* -6: one of the participants already passed intersection area
* -10: Ego and adversary are same entity

## Post encroachment-time PET
* -1: no intersection between vehicles
* -2: both participants are at intersection area at the same time (the entering or leaving times have issues. this 
happens for example if a car following scenario happens: both vehicles have entered the intersection but didn't leave) 
* -3: intersecting angle is too small for being an intersection scenario
* -4: intersecting area too long/big
* -10 Ego and adversary are same entity

## Encroachment-time ET
* -1: no intersection between vehicles
* -2: both participants are at intersection area at the same time (the entering or leaving times have issues. this 
happens for example if a car following scenario happens: both vehicles have entered the intersection but didn't leave) 
* -3: intersecting angle is too small for being an intersection scenario
* -4: intersecting area too long/big
* -10 Ego and adversary are same entity

## Predictive TTC PTTC
* -1: No collision because of velocity (front vehicle is faster)
* -3: intersecting angle is too big for being a car following scenario
* -10: Ego and adversary are same entity

## Time-to-collision TTC
* -2: No collision because of velocity (front vehicle is faster)
* -3: intersecting angle is too big for being a car following scenario
* -10: Ego and adversary are same entity

## Trajectory distance
* -6: one of the participants already passed intersection area
* -10: Ego and adversary are same entity
* 
## Worst Time-to-collision WTTC
* -1: Ego and adversary are not moving and system of equations cannot be solved
* -10: Ego and adversary are same entity