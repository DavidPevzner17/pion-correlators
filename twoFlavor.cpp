#include <iostream>
#include <cmath>
#include <random>
#include <complex>
#include <omp.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
using namespace std;
using namespace Eigen;

random_device rd;
mt19937 mt(rd());
random_device rd2;
mt19937 mt2(rd2());

int const SIZE=8;
int const SIZEt=16;
int const setSize=10000;
int const n_equil = 5000;
int const n_measure=10;
int const lag=5;
int const propSize = 12*SIZE*SIZE*SIZE*SIZEt;
int const smearNum=1;
double const epsilon=0.12;
double const betaConst=5.48;
double const e = 2.718281828459045;
double const alphaConst = 0.5;
double const lattice_spacing = 0.30;
Matrix3cd link[SIZE][SIZE][SIZE][SIZEt][4];
Matrix3cd X[setSize];
complex<double> zero (0,0);
complex<double> one(1,0);
SparseMatrix<complex<double>> updiracMatrix;
SparseMatrix<complex<double>> downdiracMatrix;
BiCGSTAB<SparseMatrix<complex<double>>> upSolver;
BiCGSTAB<SparseMatrix<complex<double>>> downSolver;

void coldStart(){ //set all links to the identity matrix
    Matrix3cd identityMatrix; 
    identityMatrix << one,zero,zero,zero,one,zero,zero,zero,one;
    for(int x=0;x<SIZE;x++){
        for(int y=0;y<SIZE;y++){
            for(int z=0;z<SIZE;z++){
                for(int t=0;t<SIZEt;t++){
                    for(int d=0;d<4;d++){
                        link[x][y][z][t][d]=identityMatrix;
                    }
                }
            }
        }
    }
    return;
}

Matrix2cd SU2_generator(void){
    uniform_real_distribution<double> dist(-0.5, 0.5);
    double r[4];
    for(int i=0;i<4;i++){
        r[i] = dist(mt);
    }
    double x0 = pow(1-epsilon*epsilon,0.5); //got rid of sgn function because it gives negative identity matrix
    double x[3];
    double mag = pow(pow(r[0],2)+pow(r[1],2)+pow(r[2],2),0.5);
    for(int i=0;i<3;i++){
        x[i] = epsilon*r[i]/mag;
    }
    Matrix2cd SU2;
    SU2(0,0)= complex<double> (x0,x[2]);
    SU2(0,1) = complex<double> (x[1],x[0]);
    SU2(1,0) = complex<double> (-x[1],x[0]);
    SU2(1,1) = complex<double> (x0,-x[2]);
    return SU2;
}

Matrix3cd SU3_generator(void){
    Matrix2cd SU2[3];
    Matrix3cd R;
    Matrix3cd S;
    Matrix3cd T;
    for(int i=0;i<3;i++){
        SU2[i] = SU2_generator();
    }
    R << SU2[0](0,0), SU2[0](0,1), zero, SU2[0](1,0), SU2[0](1,1), zero, zero, zero, one;
    S << SU2[1](0,0),zero,SU2[1](0,1),zero,one,zero,SU2[1](1,0),zero,SU2[1](1,1);
    T << one,zero,zero,zero,SU2[2](0,0),SU2[2](0,1),zero,SU2[2](1,0),SU2[2](1,1);
    return R*S*T;
}

void generate_X(){
    for(int i=0;i<setSize;i=i+2){
        Matrix3cd SU3 = SU3_generator();
        Matrix3cd SU3_hermitian = SU3.adjoint();
        X[i]=SU3;
        X[i+1]=SU3_hermitian;
    }
}

int mod(int x, int m){
    return (x % m + m) % m; // Handles negative values safely
}

Matrix3cd staple_calc(int nx, int ny, int nz, int nt, int direction){
    Matrix3cd staple, U1, U2, U3, staple1, U4, U5, U6, staple2;
    staple << zero,zero,zero,zero,zero,zero,zero,zero,zero;
    int b[4];
    int a[4];
    a[0]=0;
    a[1]=0;
    a[2]=0;
    a[3]=0;
    a[direction]=1;
    for(int nu=0;nu<4;nu++){
        b[0]=0;
        b[1]=0;
        b[2]=0;
        b[3]=0;
        b[nu]=1;
        if(nu != direction){
            // First staple term (upper staple)
            U1 = link[mod(nx + a[0], SIZE)][mod(ny + a[1], SIZE)][mod(nz + a[2], SIZE)][mod(nt + a[3], SIZEt)][nu];
            U2 = (link[mod(nx + b[0], SIZE)][mod(ny + b[1], SIZE)][mod(nz + b[2], SIZE)][mod(nt + b[3], SIZE)][direction]).adjoint();
            U3 = (link[nx][ny][nz][nt][nu]).adjoint();

            staple1 = U1*U2*U3;

            // Second staple term (lower staple)
            U4 = (link[mod(nx + a[0] - b[0], SIZE)][mod(ny + a[1] - b[1], SIZE)][mod(nz + a[2] - b[2], SIZE)][mod(nt + a[3] - b[3], SIZEt)][nu]).adjoint();
            U5 = (link[mod(nx - b[0], SIZE)][mod(ny - b[1], SIZE)][mod(nz - b[2], SIZE)][mod(nt - b[3], SIZEt)][direction]).adjoint();
            U6 = link[mod(nx - b[0], SIZE)][mod(ny - b[1], SIZE)][mod(nz - b[2], SIZE)][mod(nt - b[3], SIZEt)][nu];

            staple2 = U4*U5*U6;

            // Add to total staple
            staple = staple + staple1 + staple2;
        }
    }
    return staple;
}

void equilibriate(int nx, int ny, int nz, int nt, int direction){//update link[][][][][] matrix with Metropolis Algorithm
    //step1: generate random SU(3) matrix from set X
    uniform_int_distribution<int> dist(0, setSize-1);
    int x = dist(mt);
    Matrix3cd SU3 = X[x];

    static int total = 0;
    static int accepted = 0;
    total++;

    //step2: compute deltaS based off staple sums(hard part)
    Matrix3cd linkPrime = SU3*link[nx][ny][nz][nt][direction];
    Matrix3cd staple =  staple_calc(nx,ny,nz,nt,direction);
    double deltaS = -betaConst/3 * (((linkPrime-link[nx][ny][nz][nt][direction])*staple).trace()).real();

    //step3: generate random number r from 0 to 1. Compare to e^(-deltaS)
    uniform_real_distribution<double> dist2(0, 1);
    double r = dist2(mt2);
    double exp =pow(e,-deltaS);
    if(exp>=r){//accept
        link[nx][ny][nz][nt][direction] = linkPrime;
        accepted++;
    }
    if (total % 10000 == 0) {
        //cout << "Acceptance rate: " << (double)accepted / total << endl;
    }

    //else:reject
}

void sweep(void){
    for(int x=0;x<SIZE;x++){
        for(int y=0;y<SIZE;y++){
            for(int z=0;z<SIZE;z++){
                for(int t=0;t<SIZEt;t++){
                    for(int d=0;d<4;d++){
                        equilibriate(x,y,z,t,d);
                    }
                }
            }
        }
    }
}

Matrix3cd perp_staple_calc(int nx,int ny,int nz,int nt,int direction){
    Matrix3cd staple; 
    staple << zero,zero,zero,zero,zero,zero,zero,zero,zero;
    Matrix3cd U1,U2,U3,staple1,U4,U5,U6,staple2;
    int b[4];
    int a[4];
    a[0]=0;
    a[1]=0;
    a[2]=0;
    a[3]=0;
    a[direction]=1;
    for(int nu=0;nu<4;nu++){
        b[0]=0;
        b[1]=0;
        b[2]=0;
        b[3]=0;
        b[nu]=1;
        if(nu != direction){
            // First staple term (upper staple)
            U1 = link[nx][ny][nz][nt][nu];
            U2 = link[mod(nx+b[0],SIZE)][mod(ny+b[1],SIZE)][mod(nz+b[2],SIZE)][mod(nt+b[3],SIZEt)][direction];
            U3 = (link[mod(nx+a[0],SIZE)][mod(ny+a[1],SIZE)][mod(nz+a[2],SIZE)][mod(nt+a[3],SIZEt)][nu]).adjoint();

            staple1 = U1*U2*U3;

            // Second staple term (lower staple)
            U4 = (link[mod(nx-b[0],SIZE)][mod(ny-b[1],SIZE)][mod(nz-b[2],SIZE)][mod(nt-b[3],SIZEt)][nu]).adjoint();
            U5 = link[mod(nx-b[0],SIZE)][mod(ny-b[1],SIZE)][mod(nz-b[2],SIZE)][mod(nt-b[3],SIZEt)][direction];
            U6 = link[mod(nx-b[0]+a[0],SIZE)][mod(ny-b[1]+a[1],SIZE)][mod(nz-b[2]+a[2],SIZE)][mod(nt-b[3]+a[3],SIZEt)][nu];

            staple2 = U4*U5*U6;

            // Add to total staple
            staple = staple+ staple1+ staple2;
        }
    }
    return staple;
}

Matrix3cd project_SU3(Matrix3cd V){
    Vector3cd v1, v2;
    v1 << V(0,0), V(1,0), V(2,0);
    v2 << V(0,1), V(1,1), V(2,1);

    Vector3cd u1 = v1/v1.norm();
    Vector3cd w2 = v2- u1*u1.dot(v2);
    Vector3cd u2 = w2/w2.norm();
    Vector3cd u3 = u1.cross(u2);

    // Final matrix: u1, u2, u3 as columns
    Matrix3cd U;
    U << u1(0),u2(0),u3(0),u1(1),u2(1),u3(1),u1(2),u2(2),u3(2);
    return U;
}

void APE_smear(void){//smear global link variable
    Matrix3cd staple;
    Matrix3cd v;
    static Matrix3cd newLink[SIZE][SIZE][SIZE][SIZEt][4];
    for(int x=0;x<SIZE;x++){
        for(int y=0;y<SIZE;y++){
            for(int z=0;z<SIZE;z++){
                for(int t=0;t<SIZEt;t++){
                    for(int mu=0;mu<4;mu++){
                        staple = perp_staple_calc(x,y,z,t,mu);
                        v=(1.0-alphaConst)*link[x][y][z][t][mu]+(alphaConst/6.0)*staple;
                        newLink[x][y][z][t][mu]=project_SU3(v);
                    }
                }
            }
        }
    }
    for(int x=0;x<SIZE;x++){
        for(int y=0;y<SIZE;y++){
            for(int z=0;z<SIZE;z++){
                for(int t=0;t<SIZEt;t++){
                    for(int mu=0;mu<4;mu++){
                        link[x][y][z][t][mu]=newLink[x][y][z][t][mu];
                    }
                }
            }
        }
    }
}

int getIndex(Vector4d m, int color, int spin){
    return 12*SIZE*SIZE*SIZEt*m(0)+12*SIZE*SIZEt*m(1)+12*SIZEt*m(2)+12*m(3)+color*4+spin;
}

VectorXcd pointSource(int b0, int beta0, Vector4d m){
    VectorXcd source(propSize);
    source.setZero();
    int i = getIndex(m,b0,beta0);
    source(i)=1.0;
    return source;
}

SparseMatrix<complex<double>> dirac_operator(double fermionMass){
    SparseMatrix<complex<double>> dirac(propSize,propSize);
    Matrix4cd I = Matrix4cd:: Identity();
    Matrix4cd gamma[4];
    gamma[0] << 0,0,0,complex<double>(0,-1),0,0,complex<double>(0,-1),0,0,complex<double>(0,1),0,0,complex<double>(0,1),0,0,0;
    gamma[1] << 0,0,0,-1,0,0,1,0,0,1,0,0,-1,0,0,0;
    gamma[2] << 0,0,complex<double>(0,-1),0,0,0,0,complex<double>(0,1),complex<double>(0,1),0,0,0,0,complex<double>(0,-1),0,0;
    gamma[3] << 0,0,1,0,0,0,0,1,1,0,0,0,0,1,0,0;
    typedef Triplet<complex<double>> T;
    vector<T> tripletList;
    tripletList.reserve(9*propSize); //1 diagonal + 8 neighbors

    int x[4];
    int i,j,dx,dy,dz,dt;
    complex<double> val;
    for(x[0]=0;x[0]<SIZE;x[0]++){
        for(x[1]=0;x[1]<SIZE;x[1]++){
            for(x[2]=0;x[2]<SIZE;x[2]++){
                for(x[3]=0;x[3]<SIZEt;x[3]++){
                    for(int a=0;a<3;a++){
                        for(int alpha=0;alpha<4;alpha++){
                            i=getIndex(Vector4d(x[0],x[1],x[2],x[3]),a,alpha);
                            tripletList.push_back(T(i,i,fermionMass+4/lattice_spacing));
                            for(int mu=0;mu<4;mu++){
                                dx = (mu==0);
                                dy = (mu==1);
                                dz = (mu==2);
                                dt = (mu==3);

                                int xp = mod(x[0] + dx, SIZE);
                                int yp = mod(x[1] + dy, SIZE);
                                int zp = mod(x[2] + dz, SIZE);
                                int tp = mod(x[3] + dt, SIZEt);
                                for(int b=0;b<3;b++){
                                    for(int beta=0;beta<4;beta++){
                                        j=getIndex(Vector4d(xp,yp,zp,tp),b,beta);
                                        val = -0.5/lattice_spacing*link[x[0]][x[1]][x[2]][x[3]][mu](a,b)*(I-gamma[mu])(alpha,beta);
                                        if(mu == 3 && x[3]+dt>=SIZEt){
                                            val = -val;
                                        }
                                        tripletList.push_back(T(i,j,val));
                                    }
                                }
                                int xm = mod(x[0] - dx, SIZE);
                                int ym = mod(x[1] - dy, SIZE);
                                int zm = mod(x[2] - dz, SIZE);
                                int tm = mod(x[3] - dt, SIZEt);
                                for(int b=0;b<3;b++){
                                    for(int beta=0;beta<4;beta++){
                                        j=getIndex(Vector4d(xm,ym,zm,tm),b,beta);
                                        val = -0.5/lattice_spacing*(link[xm][ym][zm][tm][mu]).adjoint()(a,b)*(I+gamma[mu])(alpha,beta);
                                        if(mu == 3 && x[3]-dt<0){
                                            val = -val;
                                        }
                                        tripletList.push_back(T(i,j,val));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    dirac.setFromTriplets(tripletList.begin(),tripletList.end());
    return dirac;
}

void prepare_updirac_solver(double fermionMass){
    updiracMatrix = dirac_operator(fermionMass);
    upSolver.compute(updiracMatrix);
    if (upSolver.info() != Success) {
        cerr << "solver preparation failed" << endl;
        exit(1);
    }
}

void prepare_downdirac_solver(double fermionMass){
    downdiracMatrix = dirac_operator(fermionMass);
    downSolver.compute(downdiracMatrix);
    if (downSolver.info() != Success) {
        cerr << "solver preparation failed" << endl;
        exit(1);
    }
}

double* compute_obs(){
    double* sum = new double[SIZEt];
    for(int i=0;i<SIZEt;i++){
        sum[i]=0;
    }
    int* idx = new int[SIZEt];
    Vector4d origin;
    origin << 0,0,0,0;
    VectorXcd Gup(propSize);
    VectorXcd Gdown(propSize);
    VectorXcd source(propSize);
    for(int spin=0; spin<4; spin++){
        for(int color=0; color<3; color++){
            source = pointSource(color, spin, origin);
            Gup = upSolver.solve(source); // the time-consuming step
            Gdown = downSolver.solve(source);
            for(int x=0; x<SIZE; x++){
                for(int y=0; y<SIZE; y++){
                    for(int z=0; z<SIZE; z++){
                        for(int alpha=0;alpha<4;alpha++){
                            for(int a=0;a<3;a++){
                                for(int t=0;t<SIZEt;t++){
                                    idx[t]=getIndex(Vector4d(x,y,z,t),a,alpha);
                                    sum[t]=sum[t] - (Gup(idx[t])*Gdown.adjoint()(idx[t])).real();
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return sum;
}

double* compute_cnt(double upMass, double downMass){ //pion correlator calculation
    double* cnt = new double[SIZEt];
    static Matrix3cd saved_link[SIZE][SIZE][SIZE][SIZEt][4];
    for(int i=0;i<SIZEt;i++){
        cnt[i]=0;
    }
    double* obs = new double[SIZEt];
    for(int i=0;i<n_measure;i++){ //measurment phase
        for(int j=0;j<lag;j++){
            sweep(); //generate new state distributed according to the desired probability distribution
        }
        for(int x=0;x<SIZE;x++){
            for(int y=0;y<SIZE;y++){
                for(int z=0;z<SIZE;z++){
                    for(int t=0;t<SIZEt;t++){
                        for(int mu=0;mu<4;mu++){
                            saved_link[x][y][z][t][mu]=link[x][y][z][t][mu]; //copy link into saved_link
                        }
                    }
                }
            }
        }
        for(int s=0;s<smearNum;s++){
            APE_smear();
        }
        prepare_updirac_solver(upMass);
        prepare_downdirac_solver(downMass);
        obs = compute_obs();
        for(int t=0;t<SIZEt;t++){
            cnt[t]=cnt[t]+obs[t]/n_measure;
        }
        for(int x=0;x<SIZE;x++){
            for(int y=0;y<SIZE;y++){
                for(int z=0;z<SIZE;z++){
                    for(int t=0;t<SIZEt;t++){
                        for(int mu=0;mu<4;mu++){
                            link[x][y][z][t][mu]=saved_link[x][y][z][t][mu]; //copy saved_link into link
                        }
                    }
                }
            }
        }
    }
    delete[] obs;
    return cnt;
}

int main(void){
    coldStart();
    generate_X();
    for(int i=0;i<n_equil;i++){ //equilibriate system
        sweep();
    }
    double* x;
    x = compute_cnt(32,32); //input up and down quark mass here
    for(int t=0;t<SIZEt;t++){
        cout << "(" << t << "," << x[t]/x[0] << "),";
    }
 
    //g++ -I"c:/Eigen" -O3 -march=native -funroll-loops -fopenmp -ffast-math -ftree-vectorize twoFlavor.cpp -o twoFlavor.exe; ./twoFlavor.exe

    return 0;
}
