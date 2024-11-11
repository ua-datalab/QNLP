# nov 11th 2024
 
import lambeq
from lambeq.backend.grammar import Id, Ty, Box

"""#just categories
1. has objects
2. has morphisms
3. morphisms can be chained->> as long as domain and codomain matches
4. morphisms chaining is associative
5. each object has an identity morphism/FUNCTION- it is not a box, but a diagram,
 because it is a function which takes A to A.
 i.e it takes input as a function and leaves the function as is.
"""

#has objects
A = Ty('A')
B = Ty('B')
C = Ty('C')
D = Ty('D')

# 2. has morphisms
f = Box('f', dom= A, cod=B)
g = Box('g', dom= B, cod=C)
h = Box('h', dom= C, cod=D)

# 3. morphisms can be chained as long as domain and codomain matches
m = f>>g
assert f.cod == g.dom
print(f"type of m {type(m)}") #COmbinations/chained functions are called diagrams

#4.function chaining is associative
q = f >> (g >> h)
# q.draw()
r = (f >> g) >> h
# r.draw()
assert q == r

"""#5. each Object has a corresponding function called Identify function- 
written as 1A. This is like 
1 for math integer multiplication. But remember it is in function land.
so if a function hits 1A, it remains as is.
"""
oneA = Id(A)
print(f"type(oneA) is {type(oneA)}")
print(f"domain of oneA is {(oneA.dom)}")
print(f"codomain of oneA is {(oneA.cod)}")
print(f"domain of f is {(f.dom)}")
print(f"codomain of f is {(f.cod)}")

newf =  oneA >> f
print(newf)
print(f"type(newf) is {type(newf)}") #should be diagram, and value must be f itself
print(f"type(f) is {type(f)}")

#reverseing a diagram performs the dagger operation
# assert newf == f.to_diagram()
# newf.to_diagram().draw()
revF= newf[::-1]
# revF.draw()
# f.to_diagram().draw()
#but we cannot reverse a box
# revF2=f[::-1] #error
revF2=f.to_diagram()[::-1] #however if you transform f to a diagram (say you hit it with 1A) then you can reverse it.
revF2.draw()


