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
composites_into_a_function.draw()


