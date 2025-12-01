When i was writing my own solver i quickly run into problem of abstracting 
classes for the implementation. It gave me the idea that concepts like field 
(fluid for instance) is itself an abstraction and therefore i should use some 
physics library. None of the available libraries matched my use case, so here is my
attempt on this.

Suppose i am solving simple steady diffusion equation with constant diffusion coefficient
    lambda d^2/dxdx F + f = 0
Here the F is my resulting field and f is the a constant field of source terms.
we also have some scalar and lambda operator. i figured that it should not really matter
how are those fields discretized the solution method should remain the same. Let us
take a look at transient case:
    lambda d^2/dxdx F + f = dF/dt
Now we add one more dimension which is time, but in general the equation remains the same.
My idea is - "Maybe its possible to abstract out a general library for field operations".

I quickly started sketching some abstract operators like laplace, abstract classes like fields, but i 
quickly faced the problem of what should be created first. Now that i think about it, it should be our
space. We have some PDE case here or maybe even just Equation. those could be written into system then.
Back to the example, lets define Eq with space of 1 dimension (x) for steady case.

mesh = meshTool.mesh(geo, domainDescritpion) (points, elements etc)
FemDiscretization = FEMDiscretization(mesh, order = 2)

EquationSpace(dimensions = 1)
EquationSpace.setDomain(mesh) / discretization ???

F = EquationSpace->NewScalarField(init_value = 0)
F.setDiscretization(FemDiscretization)
f = EquationSpace->NewScalarFieldConstant(init_value = 10)
rhs = EquationSpace->NewScalarFieldConstant(init_value = 0)

leftBC = EquationSpace->bcs->DirichletBC(leftDomainId, value = 0)
rightBC = EquationSpace->bcs->DirichletBC(rightDomainId, value = 10)

F.setCondition(leftBc)
F.setCondition(rightBc)

equation = EquationSpace.LSE(field = F)
equation.A = F.operators().laplace()
equation.b = -f

LESSolve(equation)

thats how i would imagine my code. the internals of solution and methods required
for matrix assembly should be inside Field and discretization.

Lets take a look of how would that look like if we also applied time

mesh = meshTool.mesh(geo, domainDescritpion) (points, elements etc)
FemDiscretization = FEMDiscretization(mesh, order = 2)
TimeDiscretization = OneDRangeDiscretization(0, 100, step = 0.1)

EquationSpace(dimensions = 1)
EquationSpace.setDomain(dimension = 1, mesh)

F = EquationSpace->NewScalarField(init_value = 0)
F.setDiscretization(dimension=1, FemDiscretization)

lambda = EquationSpace->NewScalar(0.01)
f = EquationSpace->NewScalarFieldConstant(init_value = 10)
rhs = EquationSpace->NewScalarFieldConstant(init_value = 0)

leftBC = EquationSpace->bcs->DirichletBC(leftDomainId, value = 0)
rightBC = EquationSpace->bcs->DirichletBC(rightDomainId, value = 10)

F.setCondition(leftBc)
F.setCondition(rightBc)

<!-- lambda d^2/dxdx F + f = 0 -->

A = F.operator.laplace()
b = -f
LESSolve(field = F, A, b)

<!-- lambda d^2/dxdx F + f = dF/dt -->
T = F.operator.timeDer(order = 1, scheme = 'Euler')

lhs: Term = lambda * F.operator.laplace + f
rhs: Term = T

TransientLinearSolve(field = F, lhs, rhs)

So that is my general sketch for solvers. Now i came to conclusion that by using
those simple signatures that contain LHS and RHS i can solve only linear equations.
That is not a good information but it is an information regardless. I can use that
to create a library. It would be a library for linear equations including PDEs.

Now lets move on to what was the ultimate goal of this entire project. To solving N-S equations. I will try to utilize this pseudocode library
to solve euler N-S using pressure-correction TMVdV.

It is composed of 3 steps:

dU/dt - ni * laplace U = -grad P + f with dirichlet conditions
then:
laplace F = 3/2dt U
p = p + fi - ni grad U
and we iterate in time. Lets see how my library would approach that.
First of all we will be solving that in manual loop, since the intermediate steps also require time  steps etc. It will be overall much
cleaner.

space = EquationSpace(dimesnion = 2 (x and y))

U = space.NewVectorField(init_value = 0)
P = space.NewScalarField(init_value = 0)
f = space.NewVectorConstantField(init_value = 0)

Here i got an interesting idea, what if i could compose the method in a
functional way, so each step could be rewritten as a method that returns
a function to computing it. Initially it sounds more complex, but i
think that this design for algorithm composition is actually how we
should approach such problems. I will try to introduce AlgorithmStep
abstraction.


step1 (we calculate U):
we have:
dUx/dt - lambda (d^2/dx^2 Ux + d^2/dy^2 Ux) = - dP/dx + fx
dUy/dt - lambda (d^2/dy^2 Uy + d^2/dy^2 Uy) = - dP/dy + fy
-> Ux, Uy

TU = U.operator.time(scheme = tmvdv=2)
LU = U.operator.laplace()
gradP = P.operator.grad()

Here i got a thought that it would be easier, and more general if we
abstracted equations to always be of form e(x,...) = 0. Its one argument
less in the solve call and i think i would have to that anyway. Also
since we are going to iterate a lot i think the equation abstract should
be also just a function, or a class with a solve function (this way, we
can assemble matrices only once in runtime and then we just pipe fields
and iterate). It all feels very functional, so much i am slowly thinking
moving on from python prototype i will try it in functional language.

so we get an eq: 
eq = VecEquation(TU - lambda * LU + gradP - f = 0) -> should create pipes
so lets plug it in
LESSolve(field = U, eq) -> should also just create pipe for updating U. 
I wonder whether returing such pipes would actually be faster. Since we 
will just assemble numerical steps across some domain and discretization
we could control parallelism easier. Okay but lets continue with the
equation.

e2q VecEquation(TU - lambda * LU + gradP - f = 0)
Step1 = LESSolve(field = U, eq)

Fi = EquationSpace.NewScalarField(init_value = 0)
Ux = U.operator.partial(dim = x)
Uy = U.operator.partial(dim = y)
LF = Fi.operator.laplace
bc = EquationSpace.bcs->NeumannCondition(domains = ..., value = 0)
Fi.applyBC(bc)
eq = Equation(LF -(3 * TU.time_step / 2) * ( Ux + Uy ))
Step2 = ESSolve(field = Fi, Step2)
Step3 = P.update(P + Fi - lambda * (Ux.evaluate + Uy.evaluate))

Now we have 3 steps for calculating transient NS without non-linear
terms. Lets try to create my Algorithm aggregate.

pressure_correction = Algorithm(
    U, Fi, P,
    Step1,
    Step2,
    Step3
)

Here i have a trouble with defining steps. We use spatial operators and
time operators. We have the initial solution in entire domain, we 
assembled all steps. But now we would have to think what should the 
algorithm do. The time discretization is like a vector [0.0, 0.1, 0.2] etc. all out fields should be equal to init_value in our space [x, y].

if i now called algorithm.step() what exactly should happen? Should all
our fields be just discretized in space? We cant really store all fields
in space and in time. Maybe through something like:

TimeStep = ConstantTimeStep(0.1)
integrator = TimeIntegrator(TimeStep, start = 0, end = 10)

U.time.saveHistory(true)
P.time.saveHistory(true)

integrator.run(
    algorithm = algorithm
)

So that would conclude my basic setup for CFD. Before continuing to 
design i should transform those 3 examples into 3 UseCases.

Before proceeding with my LinAlgae library design i should take into
consideration more use cases from other domains. Lets think about
one example for Dynamic systems modeling and for some optimization or 
even perhaps ML cases. Also i should write out cases for other CFD
solving methods for full compressible flow with energy eq.

---------- Dynamic systems case -----------
Lets take the very basic dynamic modeling case which would be the
immortal pendulum.

d^2(fi)/dt^2 + g/l * sin(fi) = 0

So we have one parameter Fi in time space. Here we dont need spatial
discretization since we only have one scalar, we usually will have
scalars here not spatial fields. It is still a differential equation.
I would solve this using time integrator and some algorithm which should
be rk4 integration? i need to assemble that first and then try to apply
it so that it fits integrator.run(algorithm)


integrator = TimeIntegrator(0.0, 100, 0.1)
space = EquationSpace()
fi = space.NewScalar()

d^2(fi)/dt^2 + g/l * sin(fi) = 0
x1 = fi
x2 = dfi/dt = dx1/dt

dx2/dt + g/l * sin(x1) = 0

x = [x1, x2]
dx/dt = [x2, -g/l * sin(x1)]

Simplest Euler is
so x (t+1) = x + dt * dx/dt 

rk4:
k1 = f(x) = dx/dt
k2 = f(x + dt/2 k1)
k3 = f(x + dt/2 k2)
k4 = f(x + dt * k3)

So for euler i will have
eulerStep = x.update(
    x.evaluate + 
   [x[1].evaluate, dt * (g/l * sin(x[0].evaluate))]
)
integrator.run(algorithm = eulerStep)

Okay is my bottom plan so far. lets try implementation -> Python -> Cpp -> Rust!