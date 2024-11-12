# nov 11th 2024
 
import numpy as np
import lambeq
from lambeq.backend.grammar import Id, Ty, Box, Cap, Cup, Word, Diagram
from lambeq.backend.drawing import draw_equation

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
# revF2.draw()

"""monoidal categories
everything same as a category but also has
1. composing of objects- using @
2. composing of morphisms- using @
3. composing of objects is associative
4. composing of morphisms is associative
5. This time there is a true Identify for Objects. i.e I@A=A
(6. The identify on morphisms 1A still remains as is like in a category-
but note there we have to chain a function with an identity function 
(e.g. Id(A)>> f == f.to_diagram())
and  not using @. 
Now/you can also do Id(Ty()) to denote a identity morphism.
Note that since Id(Ty())@ f is a composite function (as opposed to chaining)
the domain codomain match doesnt have to happen.

i.e assert Id(A) >> f == Id(Ty()) @ f == f.to_diagram() == f@Id(Ty())

Note: composing @ is not the same as chaining >>
Composing just creates a new object/fn in a single picture. whiel chaining is literally
use f first and then use g.
 """

Z= A@B
print(f"type(z) is {type(Z)}")
x = f@g
print(f"type(x) is {type(x)}")
# x.draw()
# m.draw()

#identity of objects- called Ty()
compA = Ty() @A
print(f"type(compA) is {type(compA)}")
print(f"value of (compA) is {(compA)}")

#identity of morphisms now has two avatars, Id(A) or Id(Ty())
fMorphism= Id(A) >> f

print(f"type(fMorphism) is {type(fMorphism)}")
print(f"value of (fMorphism) is {(fMorphism)}")
assert fMorphism == f.to_diagram()
# fMorphism.draw()

fMorphismv2 = Id(Ty()) @ f
print(f"type(fMorphismv2) is {type(fMorphismv2)}")
print(f"value of (fMorphismv2) is {(fMorphismv2)}")
assert fMorphismv2 == f.to_diagram()
# fMorphismv2.draw()

assert Id(A) >> f == Id(Ty()) @ f == f.to_diagram() == f@Id(Ty())


print(f"type(Id(A)) is {type(Id(A))}")
print(f"type(Id(Ty())) is {type(Id(Ty()))}")
print(f"type((Ty())) is {type((Ty()))}")

x = Box ('x', A, A)
intermediate= x @ Id(A)
print(f"type(intermediate) is {type(intermediate)}")
print(f"value of (intermediate) is {(intermediate)}")
# intermediate.draw()


y = Box('y', A@A, B)
composites_into_a_function =  intermediate >> y
print(f"type(composites_into_a_function) is {type(composites_into_a_function)}")
print(f"value of (composites_into_a_function) is {(composites_into_a_function)}")
# composites_into_a_function.draw()

#Rigid monoidal categories
adjoints= (A.l@A) @ (A@A.l)
print(f"type(adjoints) is {type(adjoints)}")
print(f"value of (adjoints) is {(adjoints)}")




cupWithRAdjoint = Cup(A,A.r) 
print(f"type(compositeWithRAdjoint) is {type(cupWithRAdjoint)}")
print(f"value of (compositeWithRAdjoint) is {(cupWithRAdjoint)}")
# compositeWithRAdjoint.draw()


capWithLAdjoint = Cap(A,A.l) 
print(f"type(compositeWithRAdjoint) is {type(capWithLAdjoint)}")
print(f"value of (compositeWithRAdjoint) is {(capWithLAdjoint)}")
# capWithLAdjoint.draw()

cupWithLAdjoint = Cup(A.l,A) 
print(f"type(compositeWithRAdjoint) is {type(cupWithLAdjoint)}")
print(f"value of (compositeWithRAdjoint) is {(cupWithLAdjoint)}")
# cupWithLAdjoint.draw()


print(f"dom of capWithLAdjoint(s) is {capWithLAdjoint.dom}")
print(f"cod of capWithLAdjoint(s) is {capWithLAdjoint.cod}")
print(f"dom of cupWithLAdjoint(s) is {cupWithLAdjoint.dom}")
print(f"cod of cupWithLAdjoint(s) is {cupWithLAdjoint.cod}")


s = capWithLAdjoint @ Id(A)  >> Id(A)@ cupWithLAdjoint
 
print(f"type(s) is {type(s)}")
print(f"value of (s) is {(s)}")
# s.draw()
# s.normal_form().draw()

#pregroups
n= Ty('n')
s= Ty('s')

# Word1 = Word('john',n) 
# Word2 = Word('likes',n.r@s@n.l) 
# Word3 = Word('mary',n) 

all_words =[Word('john',n) ,Word('likes',n.r@s@n.l) ,Word('mary',n) ]
morphisms = [(Cup,0,1), (Cup,3,4)]


d =Diagram.create_pregroup_diagram(all_words, morphisms)
df=d.normal_form()
# draw_equation(d,df)

"""just tensors (no conversion between diagram and tensor yet)"""
from lambeq.backend.tensor import Box, Dim, Cup, Cap,Diagram    

"""#dimension==tuple of dimensions. so a box of (2,2) means 
it will have 2 rows and 2 columns and the data it can accomondate should be max 1x4
e.g.[1,3,4,4]=

"""
# so a 2d array will have dim=2
#a 3 d array will have dim of 3
a= Dim(1)
b= Dim(2)
c= Dim(3)

d= a@b@c
print(f"type of (d) is {type(d)}")
print(f"value of (d) is {d=}")

print(f'{Dim(1) @ Dim(2) @ Dim(3)=}')

f = Box(name="f",dom=b,cod=c,data=[1,2,3,4,5,6])
print(f.eval())
# f.to_diagram().draw()


c = Diagram.cups(c, c).eval(dtype=np.int64)
print(c)
