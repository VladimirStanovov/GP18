#include "sample.cpp"
//#include <mpi.h>
const string currentDateTime() {
    time_t     now = time(NULL);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
    return buf;
}
int getNFreeNodes(int* NodeBusy, int world_size)
{
    int counter = 0;
    for(int i=0;i!=world_size;i++)
        counter+=NodeBusy[i];
    return world_size-counter;
}
int getNStartedTasks(vector<int> TaskFinished, int NTasks)
{
    int counter = 0;
    for(int i=0;i!=NTasks;i++)
        if(TaskFinished[i] > 0)
            counter++;
    return counter;
}
int getNFinishedTasks(vector<int> TaskFinished, int NTasks)
{
    int counter = 0;
    for(int i=0;i!=NTasks;i++)
        if(TaskFinished[i] == 2)
            counter++;
    return counter;
}
void qSort2int(double* Mass,int* Mass2, int low, int high)
{
    int i=low;
    int j=high;
    double x=Mass[(low+high)>>1];
    do
    {
        while(Mass[i]<x)    ++i;
        while(Mass[j]>x)    --j;
        if(i<=j)
        {
            double temp=Mass[i];
            Mass[i]=Mass[j];
            Mass[j]=temp;
            double temp2=Mass2[i];
            Mass2[i]=Mass2[j];
            Mass2[j]=temp2;
            i++;    j--;
        }
    } while(i<=j);
    if(low<j)   qSort2int(Mass,Mass2,low,j);
    if(i<high)  qSort2int(Mass,Mass2,i,high);
}

const int n_oper = 22;
const double leftbordercosnt = -1.0;
const double rightbordercosnt = 1.0;
const double sigmaconst = 5.0;
const int problemtype = 1;  // 0 - classification, 1 - regression
int world_rank_global = 0;
char buffer[500];
sample Samp;

const int ResTsize1 = 10;
const int ResTsize2 = 300;

double ResultsArray[ResTsize2];

struct Params
{
    int TaskN;
    int Type;
    int NVars;
    int MaxDepth;
    int NInds;
    int NGens;
    int MaxFEvals;
    int SelType;
    int CrossType;
    int MutType;
    int InitProb;
    int ReplType;
    int Tsize;
    int TsizeRepl;
    int P1Type;
    double DSFactor;
    double IncFactor;
};

class Result
{
    public:
    int Node=0;
    int Task=0;
    double ResultTable1[ResTsize1][ResTsize2];
    void Copy(Result &Res2, int ResTsize1, int ResTsize2);
    Result(){};
    ~Result(){};
};
void Result::Copy(Result &Res2, int _ResTsize1, int _ResTsize2)
{
    Node = Res2.Node;
    Task = Res2.Task;
    for(int k=0;k!=_ResTsize1;k++)
        for(int j=0;j!=_ResTsize2;j++)
        {
            ResultTable1[k][j] = Res2.ResultTable1[k][j];
        }
}

vector<int> Operations =
{
    0,  /// if
    1,  /// - x
    1,  /// x + y
    1,  /// x - y
    1,  /// x * y
    0,  /// x > y
    0,  /// x < y
    0,  /// x == y
    0,  /// x != y
    0,  /// x / y
    0,  /// x ^ y
    1,  /// ln(x)
    1,  /// exp(x)
    1,  /// sqrt(x)
    1,  /// sin(x)
    1,  /// cos(x)
    1,  /// tan(x)
    1,  /// tanh(x)
    1,  /// 1.0/(x)
    1,  /// sinh(x)
    1,  /// cosh(x)
    1   /// abs(x)
};
std::discrete_distribution<int> OperatorSelector(Operations.begin(),Operations.end());

inline int get_oper_nar(const int oper)
{
    switch(oper)
    {
        case 0: /// if-then-else
        {return 3;}
        case 1: /// - x
        {return 1;}
        case 2: /// x + y
        {return 2;}
        case 3: /// x - y
        {return 2;}
        case 4: /// x * y
        {return 2;}
        case 5: /// x > y
        {return 2;}
        case 6: /// x < y
        {return 2;}
        case 7: /// x == y
        {return 2;}
        case 8: /// x != y
        {return 2;}
        case 9: /// x / y
        {return 2;}
        case 10: /// x ^ y
        {return 2;}
        case 11: /// ln(x)
        {return 1;}
        case 12: /// exp(x)
        {return 1;}
        case 13: /// sqrt(x)
        {return 1;}
        case 14: /// sin(x)
        {return 1;}
        case 15: /// cos(x)
        {return 1;}
        case 16: /// tan(x)
        {return 1;}
        case 17: /// tanh(x)
        {return 1;}
        case 18: /// 1.0/(x)
        {return 1;}
        case 19: /// sinh(x)
        {return 1;}
        case 20: /// cosh(x)
        {return 1;}
        case 21: /// abs(x)
        {return 1;}
    }
    return 1;
}

class node
{
    public:
    node();
    node(const double new_nextprob, const double new_nx,
         const int new_depth, const int new_number, const int max_depth);
    ~node();
    double operation(double* xvals);
    int get_num();
    void set_num(const int new_number);
    void copy_tree(node* root);
    string print(char* buffer);
    void mutate(const int mutatednode, const double new_nextprob,
                const double new_nx, const int max_depth);
    void get_crossPoint(const int CrossPoint, node* &tempCP);
    double constant;
    int xnum;
    int type;
    int oper;
    int nar;
    int depth;
    int number;
    node** next;
};
int node::get_num()
{
    if(type == 2)
    {
        if(nar == 1)
            return next[0]->get_num();
        else if(nar == 2)
        {
            int num1 = next[0]->get_num();
            int num2 = next[1]->get_num();
            return max(num1,num2);
        }
        else
        {
            int num1 = next[0]->get_num();
            int num2 = next[1]->get_num();
            int num3 = next[2]->get_num();
            return max(max(num1,num2),num3);
        }
    }
    return number;
}
void node::set_num(const int new_number)
{
    number = new_number;
    //cout<<number<<endl;
    if(type == 2)
    {
        if(nar == 1)
        {
            next[0]->set_num(new_number+1);
        }
        else if(nar == 2)
        {
            next[0]->set_num(new_number+1);
            next[1]->set_num(next[0]->get_num()+1);
        }
        else
        {
            next[0]->set_num(new_number+1);
            next[1]->set_num(next[0]->get_num()+1);
            next[2]->set_num(next[1]->get_num()+1);
        }
    }
}
void node::mutate(const int mutatednode, const double new_nextprob,
                  const double new_nx, const int max_depth)
{
    if(mutatednode == number)
    {
        if(type == 2)
        {
            if(nar == 1)
                delete next[0];
            else if(nar == 2)
            {
                delete next[0];
                delete next[1];
            }
            else
            {
                delete next[0];
                delete next[1];
                delete next[2];
            }
            delete next;
        }
        //constant = Random(leftbordercosnt,rightbordercosnt);
        constant = CauchyRand(constant,sigmaconst);
        xnum = IntRandom(new_nx);
        if(new_nextprob < Random(0,1))
            type = 2;
        else
            type = IntRandom(2);
        //oper = IntRandom(n_oper);
        oper = OperatorSelector(generator_uni_i_2);
        nar = get_oper_nar(oper);
        //cout<<"node at depth "<<depth<<" and num "<<number<<endl;
        if(type == 2)
        {
            if(nar == 1)
            {
                next = new node*[1];
                next[0] = new node(new_nextprob, new_nx, depth+1, number+1, max_depth);
            }
            else if(nar == 2)
            {
                next = new node*[2];
                next[0] = new node(new_nextprob, new_nx, depth+1, number+1, max_depth);
                int next_num = next[0]->get_num();
                next[1] = new node(new_nextprob, new_nx, depth+1, next_num+1, max_depth);
            }
            else
            {
                next = new node*[3];
                next[0] = new node(new_nextprob, new_nx, depth+1, number+1, max_depth);
                int next_num = next[0]->get_num();
                next[1] = new node(new_nextprob, new_nx, depth+1, next_num+1, max_depth);
                next_num = next[1]->get_num();
                next[2] = new node(new_nextprob, new_nx, depth+1, next_num+1, max_depth);
            }
        }
    }
    else
    {
        if(type == 2)
        {
            if(nar == 1)
                next[0]->mutate(mutatednode, new_nextprob, new_nx, max_depth);
            else if(nar == 2)
            {
                next[0]->mutate(mutatednode, new_nextprob, new_nx, max_depth);
                next[1]->mutate(mutatednode, new_nextprob, new_nx, max_depth);
            }
            else
            {
                next[0]->mutate(mutatednode, new_nextprob, new_nx, max_depth);
                next[1]->mutate(mutatednode, new_nextprob, new_nx, max_depth);
                next[2]->mutate(mutatednode, new_nextprob, new_nx, max_depth);
            }
        }
    }
}
void node::get_crossPoint(const int crossPoint, node* &tempCP)
{
    if(number == crossPoint)
    {
        tempCP = this;
        return;
    }
    if(type == 2)
    {
        if(nar == 1)
        {
            next[0]->get_crossPoint(crossPoint, tempCP);
        }
        else if(nar == 2)
        {
            next[0]->get_crossPoint(crossPoint, tempCP);
            next[1]->get_crossPoint(crossPoint, tempCP);
        }
        else
        {
            next[0]->get_crossPoint(crossPoint, tempCP);
            next[1]->get_crossPoint(crossPoint, tempCP);
            next[2]->get_crossPoint(crossPoint, tempCP);
        }
    }
    return;
}
void node::copy_tree(node* root)
{
    root->constant = constant;
    root->xnum = xnum;
    root->depth = depth;
    root->number = number;
    root->type = type;
    root->oper = oper;
    root->nar = nar;
    if(type == 2)
    {
        root->next = new node*[nar];
        for(int i=0;i!=nar;i++)
        {
            root->next[i] = new node();
            next[i]->copy_tree(root->next[i]);
        }
    }
}
node::node()
{
    constant = 0;
    xnum = 0;
    depth = 0;
    number = 0;
    type = 0;
    oper = 1;
    nar = get_oper_nar(oper);
}
node::node(const double new_nextprob, const double new_nx,
           const int new_depth, const int new_number, const int max_depth)
{
    //constant = Random(leftbordercosnt,rightbordercosnt);
    constant = NormRand(0,sigmaconst);
    xnum = IntRandom(new_nx);
    depth = new_depth;
    number = new_number;
    if(depth < max_depth && new_nextprob < Random(0,1))
        type = 2;
    else
        type = IntRandom(2);
    //oper = IntRandom(n_oper);
    oper = OperatorSelector(generator_uni_i_2);
    nar = get_oper_nar(oper);
    //cout<<"node at depth "<<depth<<" and num "<<number<<endl;
    if(type == 2)
    {
        if(nar == 1)
        {
            next = new node*[1];
            next[0] = new node(new_nextprob, new_nx, depth+1, number+1, max_depth);
        }
        else if(nar == 2)
        {
            next = new node*[2];
            next[0] = new node(new_nextprob, new_nx, depth+1, number+1, max_depth);
            int next_num = next[0]->get_num();
            next[1] = new node(new_nextprob, new_nx, depth+1, next_num+1, max_depth);
        }
        else
        {
            next = new node*[3];
            next[0] = new node(new_nextprob, new_nx, depth+1, number+1, max_depth);
            int next_num = next[0]->get_num();
            next[1] = new node(new_nextprob, new_nx, depth+1, next_num+1, max_depth);
            next_num = next[1]->get_num();
            next[2] = new node(new_nextprob, new_nx, depth+1, next_num+1, max_depth);
        }
    }
}
double node::operation(double* xvals)
{
    double res;
    if(type == 0)
        return xvals[xnum];
    if(type == 1)
        return constant;
    switch(oper)
    {
        default:
        {
            return 0;
        }
        case 0: /// if
        {
            if(next[0]->operation(xvals) > 0)
                return next[1]->operation(xvals);
            else
                return next[2]->operation(xvals);
        }
        case 1: /// -()
        {
            res = -next[0]->operation(xvals);
            if(isinf(res) && res > 0)
                res = std::numeric_limits<double>::max();
            if(isinf(res) && res < 0)
                res = -std::numeric_limits<double>::max();
            if(isnan(res))
                res = 0;
            return res;
        }
        case 2: /// x + y
        {
            res = next[0]->operation(xvals) + next[1]->operation(xvals);
            if(isinf(res) && res > 0)
                res = std::numeric_limits<double>::max();
            if(isinf(res) && res < 0)
                res = -std::numeric_limits<double>::max();
            if(isnan(res))
                res = 0;
            return res;
        }
        case 3: /// x - y
        {
            res = next[0]->operation(xvals) - next[1]->operation(xvals);
            if(isinf(res) && res > 0)
                res = std::numeric_limits<double>::max();
            if(isinf(res) && res < 0)
                res = -std::numeric_limits<double>::max();
            if(isnan(res))
                res = 0;
            return res;
        }
        case 4: /// x * y
        {
            res = next[0]->operation(xvals) * next[1]->operation(xvals);
            if(isinf(res) && res > 0)
                res = std::numeric_limits<double>::max();
            if(isinf(res) && res < 0)
                res = -std::numeric_limits<double>::max();
            if(isnan(res))
                res = 0;
            return res;
        }
        case 5: /// x > y
        {
            res = next[0]->operation(xvals) > next[1]->operation(xvals);
            if(isinf(res) && res > 0)
                res = std::numeric_limits<double>::max();
            if(isinf(res) && res < 0)
                res = -std::numeric_limits<double>::max();
            if(isnan(res))
                res = 0;
            return res;
        }
        case 6: /// x < y
        {
            res = next[0]->operation(xvals) < next[1]->operation(xvals);
            if(isinf(res) && res > 0)
                res = std::numeric_limits<double>::max();
            if(isinf(res) && res < 0)
                res = -std::numeric_limits<double>::max();
            if(isnan(res))
                res = 0;
            return res;
        }
        case 7: /// x == y
        {
            res = next[0]->operation(xvals) == next[1]->operation(xvals);
            if(isinf(res) && res > 0)
                res = std::numeric_limits<double>::max();
            if(isinf(res) && res < 0)
                res = -std::numeric_limits<double>::max();
            if(isnan(res))
                res = 0;
            return res;
        }
        case 8: /// x != y
        {
            res = next[0]->operation(xvals) != next[1]->operation(xvals);
            if(isinf(res) && res > 0)
                res = std::numeric_limits<double>::max();
            if(isinf(res) && res < 0)
                res = -std::numeric_limits<double>::max();
            if(isnan(res))
                res = 0;
            return res;
        }
        case 9: /// x / y
        {
            res = next[0]->operation(xvals) / next[1]->operation(xvals);
            if(isinf(res) && res > 0)
                res = std::numeric_limits<double>::max();
            if(isinf(res) && res < 0)
                res = -std::numeric_limits<double>::max();
            if(isnan(res))
                res = 0;
            return res;
        }
        case 10: /// x ^ y
        {
            res = pow(next[0]->operation(xvals),next[1]->operation(xvals));
            if(isinf(res) && res > 0)
                res = std::numeric_limits<double>::max();
            if(isinf(res) && res < 0)
                res = -std::numeric_limits<double>::max();
            if(isnan(res))
                res = 0;
            return res;
        }
        case 11: /// ln(x)
        {
            res = log(next[0]->operation(xvals));
            if(isinf(res) && res > 0)
                res = std::numeric_limits<double>::max();
            if(isinf(res) && res < 0)
                res = -std::numeric_limits<double>::max();
            if(isnan(res))
                res = 0;
            return res;
        }
        case 12: /// exp(x)
        {
            res = exp(next[0]->operation(xvals));
            if(isinf(res) && res > 0)
                res = std::numeric_limits<double>::max();
            if(isinf(res) && res < 0)
                res = -std::numeric_limits<double>::max();
            if(isnan(res))
                res = 0;
            return res;
        }
        case 13: /// sqrt(x)
        {
            res = sqrt(next[0]->operation(xvals));
            if(isinf(res) && res > 0)
                res = std::numeric_limits<double>::max();
            if(isinf(res) && res < 0)
                res = -std::numeric_limits<double>::max();
            if(isnan(res))
                res = 0;
            return res;
        }
        case 14: /// sin(x)
        {
            res = sin(next[0]->operation(xvals));
            if(isinf(res) && res > 0)
                res = std::numeric_limits<double>::max();
            if(isinf(res) && res < 0)
                res = -std::numeric_limits<double>::max();
            if(isnan(res))
                res = 0;
            return res;
        }
        case 15: /// cos(x)
        {
            res = cos(next[0]->operation(xvals));
            if(isinf(res) && res > 0)
                res = std::numeric_limits<double>::max();
            if(isinf(res) && res < 0)
                res = -std::numeric_limits<double>::max();
            if(isnan(res))
                res = 0;
            return res;
        }
        case 16: /// tan(x)
        {
            res = tan(next[0]->operation(xvals));
            if(isinf(res) && res > 0)
                res = std::numeric_limits<double>::max();
            if(isinf(res) && res < 0)
                res = -std::numeric_limits<double>::max();
            if(isnan(res))
                res = 0;
            return res;
        }
        case 17: /// tanh(x)
        {
            res = tanh(next[0]->operation(xvals));
            if(isinf(res) && res > 0)
                res = std::numeric_limits<double>::max();
            if(isinf(res) && res < 0)
                res = -std::numeric_limits<double>::max();
            if(isnan(res))
                res = 0;
            return res;
        }
        case 18: /// 1.0/(x)
        {
            res = 1.0/(next[0]->operation(xvals));
            if(isinf(res) && res > 0)
                res = std::numeric_limits<double>::max();
            if(isinf(res) && res < 0)
                res = -std::numeric_limits<double>::max();
            if(isnan(res))
                res = 0;
            return res;
        }
        case 19: /// sinh(x)
        {
            res = sinh(next[0]->operation(xvals));
            if(isinf(res) && res > 0)
                res = std::numeric_limits<double>::max();
            if(isinf(res) && res < 0)
                res = -std::numeric_limits<double>::max();
            if(isnan(res))
                res = 0;
            return res;
        }
        case 20: /// cosh(x)
        {
            res = cosh(next[0]->operation(xvals));
            if(isinf(res) && res > 0)
                res = std::numeric_limits<double>::max();
            if(isinf(res) && res < 0)
                res = -std::numeric_limits<double>::max();
            if(isnan(res))
                res = 0;
            return res;
        }
        case 21: /// abs(x)
        {
            res = abs(next[0]->operation(xvals));
            if(isinf(res) && res > 0)
                res = std::numeric_limits<double>::max();
            if(isinf(res) && res < 0)
                res = -std::numeric_limits<double>::max();
            if(isnan(res))
                res = 0;
            return res;
        }
    }
}
string node::print(char* buffer)
{
    if(type == 0)
    {
        sprintf(buffer,"x[%d]",xnum);
        string tmp = buffer;
        return tmp;
    }
    if(type == 1)
    {
        sprintf(buffer,"(%f)",constant);
        string tmp = buffer;
        return tmp;
    }
    switch(oper)
    {
        default:
        {
            return "";
        }
        case 0:
        {
            return "(if(" + next[0]->print(buffer) + "_>_0)_then_(" +
                    next[1]->print(buffer) + ")_else_(" +
                    next[2]->print(buffer) + "))";
        }
        case 1: /// -()
        {return "(-" + next[0]->print(buffer) + ")";}
        case 2: /// x + y
        {return "(" + next[0]->print(buffer) + "+" + next[1]->print(buffer) + ")";}
        case 3: /// x - y
        {return "(" + next[0]->print(buffer) + "-" + next[1]->print(buffer) + ")";}
        case 4: /// x * y
        {return "(" + next[0]->print(buffer) + "*" + next[1]->print(buffer) + ")";}
        case 5: /// x > y
        {return "(" + next[0]->print(buffer) + "_>_" + next[1]->print(buffer) + ")";}
        case 6: /// x < y
        {return "(" + next[0]->print(buffer) + "_<_" + next[1]->print(buffer) + ")";}
        case 7: /// x == y
        {return "(" + next[0]->print(buffer) + "==" + next[1]->print(buffer) + ")";}
        case 8: /// x != y
        {return "(" + next[0]->print(buffer) + "!=" + next[1]->print(buffer) + ")";}
        case 9: /// x / y
        {return "(" + next[0]->print(buffer) + "/" + next[1]->print(buffer) + ")";}
        case 10: /// x ^ y
        {return "(" + next[0]->print(buffer) + "^" + next[1]->print(buffer) + ")";}
        case 11: /// ln(x)
        {return "log(" + next[0]->print(buffer) + ")";}
        case 12: /// exp(x)
        {return "exp(" + next[0]->print(buffer) + ")";}
        case 13: /// sqrt(x)
        {return "sqrt(" + next[0]->print(buffer) + ")";}
        case 14: /// sin(x)
        {return "sin(" + next[0]->print(buffer) + ")";}
        case 15: /// cos(x)
        {return "cos(" + next[0]->print(buffer) + ")";}
        case 16: /// tan(x)
        {return "tan(" + next[0]->print(buffer) + ")";}
        case 17: /// tanh(x)
        {return "tanh(" + next[0]->print(buffer) + ")";}
        case 18: /// 1.0/(x)
        {return "1.0/(" + next[0]->print(buffer) + ")";}
        case 19: /// sinh(x)
        {return "sinh(" + next[0]->print(buffer) + ")";}
        case 20: /// cosh(x)
        {return "cosh(" + next[0]->print(buffer) + ")";}
        case 21: /// abs(x)
        {return "abs(" + next[0]->print(buffer) + ")";}
    }
}
node::~node()
{
    if(type == 2)
    {
        if(nar == 3)
        {
            delete next[2];
            delete next[1];
        }
        if(nar == 2)
            delete next[1];
        delete next[0];
        delete next;
    }
    //delete this;
}
class Forest
{
    public:
    Forest(const int new_NVars, const int new_MaxDepth, const int new_NInds,
               const int new_NGens, const int new_MaxFEvals, const int new_SelType,
               const int new_CrossType, const int new_MutType, const double new_InitProb,
               sample &Samp, const int new_FoldOnTest, const int new_ReplType, const int new_Tsize,
               const double new_DSFactor, const double new_IncFactor, const int P1Type,
               const int new_max_length, const int new_TsizeRepl);
    ~Forest();
    node** Popul;
    node** PopulTemp;
    node* BestInd;
    int* Indexes;
    int* OrderCases;
    int* IndActive;
    int* InstanceActive;
    int* InstanceIndex;
    int* NFESteps;
    int* AlreadyChosen;
    double* InstanceWeight;
    double* InstanceRecCounter;
    double* InstanceProb;
    double* FitMass;
    double* FitTemp;
    double** Perf;
    double** PerfTemp;
    double* SortPerf;
    double* FitMassCopy;
    double* MedAbsDist;
    double* MedAbsDistEps;
    double* MedAbsDistCopy;
    vector<double> FitTemp_vec;
    vector<double> InstProb;
    double LenWeight = 1e-8;
    int PerfSize;
    int PerfSizeDS;
    int ReplType;
    int Tsize;
    int TsizeRepl;
    int NVars;
    int MaxDepth;
    int NInds;
    int NGens;
    int MaxFEvals;
    int NFEvals;
    int SelType;
    int CrossType;
    int MutType;
    int P1Type;
    int FoldOnTest;
    int bestNum;
    int LastStep;
    int max_length;
    double IncFactor;
    double InitProb;
    double bestFit;
    double bestTestFit;
    double DSFactor;
    double MedMedAbsDist;
    void FitCalc(node* ind, sample &Samp, int index);
    void PerfCalcFull(node* ind, sample &Samp, int index);
    double FitCalcTest(node* ind, sample &Samp);
    void PrintPoints(node* ind, sample &Samp);
    void MainLoop(sample &Samp);
    void FindNSaveBest();
    void PrepareLexicase(int SizeMultiplier);
    void PrepareLexicaseDS(int SizeMultiplier);
    //void PrepareLexicaseIS(int SizeMultiplier);
    void GenerateDS();
    void GenerateIDS();
    void GenerateIS();
    void CreateNewPop();
    void CreateNewPopSort();
    void CreateNewPopTour();
    void CreateNewPopRank();
    void CreateNewPopLexicase();
    void CreateNewPopLexicaseDS();
    void CreateNewPopLexicaseIS();
    void CreateNewPopOffspring();
    int RankSelection();
    int LexicaseSelection();
    int LexicaseSelectionDS();
    int LexicaseSelectionIS();
    void StandardCrossover(const int Num, const int Selected);
    void Mutation(const int Num);
};

Forest::Forest(const int new_NVars, const int new_MaxDepth, const int new_NInds,
               const int new_NGens, const int new_MaxFEvals, const int new_SelType,
               const int new_CrossType, const int new_MutType, const double new_InitProb,
               sample& Samp, const int new_FoldOnTest, const int new_ReplType, const int new_Tsize,
               const double new_DSFactor, const double new_IncFactor, const int new_P1Type,
               const int new_max_length, const int new_TsizeRepl)
{
    NVars = new_NVars;
    MaxDepth = new_MaxDepth;
    NInds = new_NInds;
    NGens = new_NGens;
    MaxFEvals = new_MaxFEvals;
    NFEvals = 0;
    LastStep = 0;
    SelType = new_SelType;
    P1Type = new_P1Type;
    CrossType = new_CrossType;
    MutType = new_MutType;
    InitProb = new_InitProb;
    FoldOnTest = new_FoldOnTest;
    PerfSize = Samp.Size;
    ReplType = new_ReplType;
    Tsize = new_Tsize;
    TsizeRepl = new_TsizeRepl;
    DSFactor = new_DSFactor;
    IncFactor = new_IncFactor;
    max_length = new_max_length;

    NFESteps = new int[ResTsize2];
    for(int i=0;i!=ResTsize2;i++)
    {
        NFESteps[i] = int(double(MaxFEvals)*double(i+1)/double(ResTsize2)*3.0);
    }
    IndActive = new int[NInds*2];
    OrderCases = new int[PerfSize];
    InstanceActive = new int[PerfSize];
    InstanceIndex = new int[PerfSize];
    AlreadyChosen = new int[NInds*2];
    MedAbsDist = new double[PerfSize];
    MedAbsDistCopy = new double[PerfSize];
    InstanceWeight = new double[PerfSize];
    InstanceRecCounter = new double[PerfSize];
    InstanceProb = new double[PerfSize];
    SortPerf = new double[NInds*2];
    FitMass = new double[NInds*2];
    FitTemp = new double[NInds*2];
    Perf = new double*[NInds*2];
    PerfTemp = new double*[NInds*2];
    Indexes = new int[NInds*2];
    FitMassCopy = new double[NInds*2];
    FitTemp_vec.resize(NInds);
    InstProb.resize(PerfSize);
    Popul = new node*[NInds*2];
    PopulTemp = new node*[NInds*2];
    BestInd = new node(new_InitProb,NVars,0,0,MaxDepth);
    for(int i=0;i!=NInds;i++)
    {
        Popul[i] = new node(new_InitProb,NVars,0,0,MaxDepth);
        Popul[NInds+i] = new node(1,NVars,0,0,0);
        PopulTemp[i] = new node(1,NVars,0,0,0);
        PopulTemp[NInds+i] = new node(1,NVars,0,0,0);
        Perf[i] = new double[Samp.Size];
        Perf[NInds+i] = new double[Samp.Size];
        PerfTemp[i] = new double[Samp.Size];
        PerfTemp[NInds+i] = new double[Samp.Size];
    }
    if(SelType == 0 || SelType == 1 || SelType == 2)
    {
        PerfSizeDS = PerfSize;
        for(int i=0;i!=PerfSize;i++)
        {
            InstanceIndex[i] = i;
            InstanceRecCounter[i] = 1.0;
        }
        for(int i=0;i!=PerfSize;i++)
        {
            InstanceActive[InstanceIndex[i]] = 1;
            InstanceWeight[InstanceIndex[i]] = 1;
        }
    }
    for(int i=0;i!=PerfSize;i++)
    {
        InstanceRecCounter[i] = 1.0;
    }
}

void Forest::FitCalc(node* ind, sample &Samp, int index)
{
    double Error = 0;
    double Error2 = 0;
    double Mean = 0;
    for(int i=0;i!=PerfSize;i++)
    {
        if(InstanceActive[i] == 1)
        {
            if(Samp.GetCVFoldNum(i) != 1)
            {
                Mean += Samp.Outputs[i][0];
            }
        }
    }
    Mean /= double(PerfSizeDS);
    for(int i=0;i!=PerfSize;i++)
    {
        if(Samp.GetCVFoldNum(i) != 1 && InstanceActive[i] == 1)
        {
            double selected = ind->operation(Samp.Inputs[i]);
            double output = Samp.Outputs[i][0];
            Error += (selected - output)*(selected - output);
            Error2 += (Mean - selected)*(Mean - selected);
            Perf[index][i] = (selected - output)*(selected - output);
        }
    }
    Error = -(1.0 - Error/Error2);
    NFEvals++;
    if(isnan(Error))
    {
        Error = std::numeric_limits<double>::min();
    }
    FitMass[index] = Error + Popul[index]->get_num()*LenWeight;
    if(FitMass[index] < bestFit || NFEvals == 1)
    {
        bestFit = FitMass[index];
        bestNum = index;
        delete BestInd;
        BestInd = new node(1,NVars,0,0,0);
        Popul[bestNum]->copy_tree(BestInd);
        bestTestFit = FitCalcTest(BestInd, Samp);
    }
    if(NFEvals == NFESteps[LastStep])
    {
        //cout<<NFEvals<<"\t"<<LastStep<<endl;
        ResultsArray[LastStep*3+0] = -bestFit;
        ResultsArray[LastStep*3+1] = -bestTestFit;
        ResultsArray[LastStep*3+2] = BestInd->get_num();
        LastStep++;
    }
    //cout<<Error<<endl;
}

void Forest::PerfCalcFull(node* ind, sample &Samp, int index)
{
    double Mean = 0;
    for(int i=0;i!=PerfSize;i++)
    {
        if(Samp.GetCVFoldNum(i) != 1)
        {
            Mean += Samp.Outputs[i][0];
        }
    }
    Mean /= double(PerfSize);
    for(int i=0;i!=PerfSize;i++)
    {
        if(Samp.GetCVFoldNum(i) != 1)
        {
            double selected = ind->operation(Samp.Inputs[i]);
            double output = Samp.Outputs[i][0];
            Perf[index][i] = (selected - output)*(selected - output);
        }
    }
}

double Forest::FitCalcTest(node* ind, sample &Samp)
{
    double Error = 0;
    double Error2 = 0;
    double Mean = 0;
    for(int i=0;i!=Samp.Size;i++)
    {
        if(problemtype == 1)
        {
            if(Samp.GetCVFoldNum(i) == 1)
            {
                Mean += Samp.Outputs[i][0];
            }
        }
    }
    Mean /= double(Samp.Size);
    for(int i=0;i!=Samp.Size;i++)
    {
        if(Samp.GetCVFoldNum(i) == 1)
        {
            double selected = ind->operation(Samp.Inputs[i]);
            Error += (selected - Samp.Outputs[i][0])*(selected - Samp.Outputs[i][0]);
            Error2 += (Mean - Samp.Outputs[i][0])*(Mean - Samp.Outputs[i][0]);
        }
    }
    Error = -(1.0 - Error/Error2);
    if(isnan(Error))
    {
        Error = std::numeric_limits<double>::min();
    }
    return Error;
}

void Forest::CreateNewPop()
{
    for(int i=0;i!=NInds;i++)
    {
        if(FitMass[NInds+i] < FitMass[i])
        {
            delete Popul[i];
            Popul[i] = new node(1,NVars,0,0,0);
            Popul[NInds+i]->copy_tree(Popul[i]);
            FitMass[i] = FitMass[NInds+i];
            swap(Perf[i],Perf[NInds+i]);
        }
    }
}
void Forest::CreateNewPopSort()
{
    for(int i=0;i!=NInds*2;i++)
    {
        for(int j=i;j!=NInds*2;j++)
        {
            if(FitMass[i] > FitMass[j]) //replace
            {
                swap(Popul[i],Popul[j]);
                swap(FitMass[i],FitMass[j]);
                swap(Perf[i],Perf[j]);
            }
        }
        //cout<<endl;
    }
}
void Forest::CreateNewPopTour()
{
    for(int i=0;i!=NInds*2;i++)
    {
        AlreadyChosen[i] = 0;
    }

    for(int i=0;i!=NInds;i++)
    {
        int n_same = 0;
        int besti;
        do
        {
            besti = IntRandom(NInds*2);
            double bestf = FitMass[besti];
            for(int t=1;t!=TsizeRepl;t++)
            {
                int tempi = IntRandom(NInds*2);
                if(FitMass[tempi] < bestf)
                {
                    bestf = FitMass[tempi];
                    besti = tempi;
                }
            }
            n_same++;
        } while(AlreadyChosen[besti] == 1 && n_same < 25);
        AlreadyChosen[besti] = 1;

        delete PopulTemp[besti];
        PopulTemp[besti] = new node(1,NVars,0,0,0);
        Popul[besti]->copy_tree(PopulTemp[besti]);
        FitTemp[besti] = FitMass[besti];
        for(int j=0;j!=PerfSize;j++)
        {
            PerfTemp[besti][j] = Perf[besti][j];
        }
    }

    for(int i=0;i!=NInds;i++)
    {
        delete Popul[i];
        Popul[i] = new node(1,NVars,0,0,0);
        PopulTemp[i]->copy_tree(Popul[i]);
        FitMass[i] = FitTemp[i];
        for(int j=0;j!=PerfSize;j++)
        {
            Perf[i][j] = PerfTemp[i][j];
        }
    }
}
void Forest::CreateNewPopRank()
{
    for(int i=0;i!=NInds*2;i++)
    {
        AlreadyChosen[i] = 0;
    }
    double minfit = FitMass[0];
    double maxfit = FitMass[0];
    for(int i=0;i!=NInds*2;i++)
    {
        FitMassCopy[i] = FitMass[i];
        Indexes[i] = i;
        if(FitMass[i] >= maxfit)
            maxfit = FitMass[i];
        if(FitMass[i] <= minfit)
            minfit = FitMass[i];
    }
    if(minfit != maxfit)
        qSort2int(FitMassCopy,Indexes,0,NInds*2-1);
    FitTemp_vec.resize(NInds*2);
    for(int i=0;i!=NInds*2;i++)
        FitTemp_vec[i] = exp(double(i+1)/double(NInds*2)*3);
    std::discrete_distribution<int> ComponentSelector(FitTemp_vec.begin(),FitTemp_vec.end());

    for(int i=0;i!=NInds;i++)
    {
        int n_same = 0;
        int besti;
        do
        {
            besti = ComponentSelector(generator_uni_i_2);
            n_same++;
        } while(AlreadyChosen[besti] == 1 && n_same < 25);
        AlreadyChosen[besti] = 1;

        delete PopulTemp[besti];
        PopulTemp[besti] = new node(1,NVars,0,0,0);
        Popul[besti]->copy_tree(PopulTemp[besti]);
        FitTemp[besti] = FitMass[besti];
        for(int j=0;j!=PerfSize;j++)
        {
            PerfTemp[besti][j] = Perf[besti][j];
        }
    }

    for(int i=0;i!=NInds;i++)
    {
        delete Popul[i];
        Popul[i] = new node(1,NVars,0,0,0);
        PopulTemp[i]->copy_tree(Popul[i]);
        FitMass[i] = FitTemp[i];
        for(int j=0;j!=PerfSize;j++)
        {
            Perf[i][j] = PerfTemp[i][j];
        }
    }
}
void Forest::CreateNewPopLexicase()
{
    for(int i=0;i!=NInds*2;i++)
    {
        AlreadyChosen[i] = 0;
    }

    for(int i=0;i!=NInds;i++)
    {
        int n_same = 0;
        int besti;
        do
        {
            for(int i=0;i!=PerfSize*5;i++)
            {
                swap(OrderCases[IntRandom(PerfSize)],OrderCases[IntRandom(PerfSize)]);
            }
            for(int i=0;i!=NInds*2;i++)
            {
                IndActive[i] = 1;
            }
            int CaseIndex = 0;
            int NActive = NInds*2;
            while(NActive > 1 && CaseIndex < PerfSize)
            {
                int CurCase = OrderCases[CaseIndex];
                double tempBest = Perf[0][CurCase];
                for(int i=0;i!=NInds*2;i++)
                {
                    if(Perf[i][CurCase] < tempBest)
                        tempBest = Perf[i][CurCase];
                }
                double Med = MedAbsDist[CurCase];
                for(int i=0;i!=NInds*2;i++)
                {
                    if(fabs(tempBest - Perf[i][CurCase]) > Med)
                    {
                        NActive -= IndActive[i];
                        IndActive[i] = 0;
                    }
                    if(NActive == 1)
                        break;
                }
                CaseIndex++;
            }
            int Selected = IntRandom(NActive);
            for(int i=0;i!=NInds*2;i++)
            {
                if(Selected == 0 && IndActive[i] == 1)
                {
                    Selected = i;
                    break;
                }
                Selected -= IndActive[i];
            }
            besti = Selected;
            n_same++;
        } while(AlreadyChosen[besti] == 1 && n_same < 25);
        AlreadyChosen[besti] = 1;

        delete PopulTemp[besti];
        PopulTemp[besti] = new node(1,NVars,0,0,0);
        Popul[besti]->copy_tree(PopulTemp[besti]);
        FitTemp[besti] = FitMass[besti];
        for(int j=0;j!=PerfSize;j++)
        {
            PerfTemp[besti][j] = Perf[besti][j];
        }
    }

    for(int i=0;i!=NInds;i++)
    {
        delete Popul[i];
        Popul[i] = new node(1,NVars,0,0,0);
        PopulTemp[i]->copy_tree(Popul[i]);
        FitMass[i] = FitTemp[i];
        for(int j=0;j!=PerfSize;j++)
        {
            Perf[i][j] = PerfTemp[i][j];
        }
    }
}
void Forest::CreateNewPopLexicaseDS()
{
    for(int i=0;i!=NInds*2;i++)
    {
        AlreadyChosen[i] = 0;
    }

    for(int i=0;i!=NInds;i++)
    {
        int n_same = 0;
        int besti;
        do
        {
            for(int i=0;i!=PerfSize*5;i++)
            {
                swap(OrderCases[IntRandom(PerfSize)],OrderCases[IntRandom(PerfSize)]);
            }
            for(int i=0;i!=NInds*2;i++)
            {
                IndActive[i] = 1;
            }
            int CaseIndex = 0;
            int CaseNumber = 0;
            int NActive = NInds*2;
            while(NActive > 1 && CaseNumber < PerfSize && CaseIndex < PerfSize)
            {
                int CurCase = OrderCases[CaseIndex];
                if(InstanceActive[CurCase] != 1)
                {
                    CaseIndex++;
                    if(CaseIndex >= PerfSize)
                        break;
                    continue;
                }
                double tempBest = Perf[0][CurCase];
                for(int i=0;i!=NInds*2;i++)
                {
                    if(Perf[i][CurCase] < tempBest)
                        tempBest = Perf[i][CurCase];
                }
                double Med = MedAbsDist[CurCase];
                for(int i=0;i!=NInds*2;i++)
                {
                    if(fabs(tempBest - Perf[i][CurCase]) > Med)
                    {
                        NActive -= IndActive[i];
                        IndActive[i] = 0;
                    }
                    if(NActive == 1)
                        break;
                }
                CaseIndex++;
                CaseNumber++;
            }
            int Selected = IntRandom(NActive);
            for(int i=0;i!=NInds*2;i++)
            {
                if(Selected == 0 && IndActive[i] == 1)
                {
                    Selected = i;
                    break;
                }
                Selected -= IndActive[i];
            }
            besti = Selected;
            n_same++;
        } while(AlreadyChosen[besti] == 1 && n_same < 25);
        AlreadyChosen[besti] = 1;

        delete PopulTemp[besti];
        PopulTemp[besti] = new node(1,NVars,0,0,0);
        Popul[besti]->copy_tree(PopulTemp[besti]);
        FitTemp[besti] = FitMass[besti];
        for(int j=0;j!=PerfSize;j++)
        {
            PerfTemp[besti][j] = Perf[besti][j];
        }
    }

    for(int i=0;i!=NInds;i++)
    {
        delete Popul[i];
        Popul[i] = new node(1,NVars,0,0,0);
        PopulTemp[i]->copy_tree(Popul[i]);
        FitMass[i] = FitTemp[i];
        for(int j=0;j!=PerfSize;j++)
        {
            Perf[i][j] = PerfTemp[i][j];
        }
    }
}
void Forest::CreateNewPopOffspring()
{
    int worstindex = 0;
    double worstfit = FitMass[0];
    for(int i=0;i!=NInds;i++)
    {
        delete Popul[i];
        Popul[i] = new node(1,NVars,0,0,0);
        Popul[NInds+i]->copy_tree(Popul[i]);
        FitMass[i] = FitMass[NInds+i];
        swap(Perf[i],Perf[NInds+i]);
        if(FitMass[i] < worstfit)
        {
            worstfit = FitMass[i];
            worstindex = i;
        }
    }
    delete Popul[worstindex];
    Popul[worstindex] = new node(1,NVars,0,0,0);
    BestInd->copy_tree(Popul[worstindex]);
}
void Forest::PrepareLexicase(int SizeMultiplier)
{
    for(int c=0;c!=PerfSize;c++)
    {
        for(int i=0;i!=NInds*SizeMultiplier;i++)
        {
            SortPerf[i] = Perf[i][c];
        }
        qSort1(SortPerf,0,NInds-1);
        MedAbsDist[c] = SortPerf[(NInds) >> 1];
        for(int i=0;i!=NInds*SizeMultiplier;i++)
        {
            SortPerf[i] = fabs(Perf[i][c]-MedAbsDist[c]);
        }
        qSort1(SortPerf,0,NInds-1);
        MedAbsDist[c] = SortPerf[(NInds) >> 1];
        OrderCases[c] = c;
        //cout<<c<<endl;
    }
}
void Forest::PrepareLexicaseDS(int SizeMultiplier)
{
    for(int c=0;c!=PerfSize;c++)
    {
        if(InstanceActive[c] == 1)
        {
            for(int i=0;i!=NInds*SizeMultiplier;i++)
            {
                SortPerf[i] = Perf[i][c];
            }
            qSort1(SortPerf,0,NInds-1);
            MedAbsDist[c] = SortPerf[(NInds) >> 1];
            for(int i=0;i!=NInds*SizeMultiplier;i++)
            {
                SortPerf[i] = fabs(Perf[i][c]-MedAbsDist[c]);
            }
            qSort1(SortPerf,0,NInds-1);
            MedAbsDist[c] = SortPerf[(NInds) >> 1];
            //cout<<c<<endl;
        }
        OrderCases[c] = c;
    }
}
void Forest::GenerateDS()
{
    PerfSizeDS = int(double(PerfSize)/double(DSFactor));
    for(int i=0;i!=PerfSize;i++)
    {
        InstanceIndex[i] = i;
    }
    for(int i=0;i!=PerfSize*10;i++)
    {
        swap(InstanceIndex[IntRandom(PerfSize)],InstanceIndex[IntRandom(PerfSize)]);
    }
    for(int i=0;i!=PerfSizeDS;i++)
    {
        InstanceActive[InstanceIndex[i]] = 1;
        InstanceWeight[InstanceIndex[i]] = 1;
    }
    for(int i=PerfSizeDS;i!=PerfSize;i++)
    {
        InstanceActive[InstanceIndex[i]] = 0;
        InstanceWeight[InstanceIndex[i]] = 0;
    }
}
void Forest::GenerateIDS()
{
    PerfSizeDS = int(double(PerfSize)/double(DSFactor));
    for(int i=0;i!=PerfSize;i++)
    {
        InstanceIndex[i] = i;
    }
    for(int i=0;i!=PerfSize*10;i++)
    {
        swap(InstanceIndex[IntRandom(PerfSize)],InstanceIndex[IntRandom(PerfSize)]);
    }
    for(int i=0;i!=PerfSizeDS;i++)
    {
        InstanceActive[InstanceIndex[i]] = 1;
        InstanceWeight[InstanceIndex[i]] = 1;
    }
    for(int i=PerfSizeDS;i!=PerfSize;i++)
    {
        InstanceActive[InstanceIndex[i]] = 0;
        InstanceWeight[InstanceIndex[i]] = 0;
    }
}
void Forest::GenerateIS()
{
    for(int c=0;c!=PerfSize;c++)
    {
        if(InstanceActive[c] == 1)
        {
            for(int i=0;i!=NInds;i++)
            {
                SortPerf[i] = Perf[i][c];
            }
            qSort1(SortPerf,0,NInds-1);
            MedAbsDist[c] = SortPerf[(NInds) >> 1];
            //cout<<c<<endl;
        }
        OrderCases[c] = c;
    }
    int counter = 0;
    double MeanAbsDist = 0;
    for(int c=0;c!=PerfSize;c++)
    {
        if(InstanceActive[c] == 1)
        {
            MedAbsDistCopy[counter] = MedAbsDist[c];
            MeanAbsDist += MedAbsDist[c];
            counter ++;
        }
    }
    qSort1(MedAbsDistCopy,0,counter-1);
    MedMedAbsDist = MedAbsDistCopy[(counter) >> 1];//MeanAbsDist/double(counter);//
    int n_increased = 0;
    for(int c=0;c!=PerfSize;c++)
    {
        //if(InstanceActive[c] == 1)
        {
            if(Perf[bestNum][c] <= MedMedAbsDist)
            {
                InstanceRecCounter[c] += IncFactor;
                n_increased++;
            }
            else
            {
                InstanceRecCounter[c] = 1.0;
            }
        }
    }
    //cout<<"n_increased: "<<n_increased<<endl;
    for(int c=0;c!=PerfSize;c++)
    {
        InstanceProb[c] = 1.0/InstanceRecCounter[c];
    }
    PerfSizeDS = int(double(PerfSize)/double(DSFactor));
    for(int c=0;c!=PerfSize;c++)
        InstProb[c] = InstanceProb[c];
    std::discrete_distribution<int> ComponentSelector(InstProb.begin(),InstProb.end());
    for(int i=0;i!=PerfSize;i++)
    {
        InstanceIndex[i] = -1;
        InstanceActive[i] = 0;
        InstanceWeight[i] = 0;
    }
    for(int i=0;i!=PerfSizeDS;i++)
    {
        int temp = -1;
        int nsame = 0;
        do
        {
            nsame = 0;
            temp = ComponentSelector(generator_uni_i_2);
            for(int j=0;j!=PerfSizeDS;j++)
            {
                nsame += (temp == InstanceIndex[j]);
            }
        }
        while(nsame > 0);
        InstanceIndex[i] = temp;
    }
    for(int i=0;i!=PerfSizeDS;i++)
    {
        InstanceActive[InstanceIndex[i]] = 1;
        InstanceWeight[InstanceIndex[i]] = 1;
    }
}
void Forest::FindNSaveBest()
{
    bestFit = FitMass[0];
    bestNum = 0;
    for(int i=0;i!=NInds;i++)
    {
        if(FitMass[i] < bestFit)
        {
            bestFit = FitMass[i];
            bestNum = i;
        }
    }
    delete BestInd;
    BestInd = new node(1,NVars,0,0,0);
    Popul[bestNum]->copy_tree(BestInd);
}

int Forest::RankSelection()
{
    return IntRandom(NInds);
}
int Forest::LexicaseSelection()
{
    for(int i=0;i!=PerfSize*5;i++)
    {
        swap(OrderCases[IntRandom(PerfSize)],OrderCases[IntRandom(PerfSize)]);
    }
    for(int i=0;i!=NInds;i++)
    {
        IndActive[i] = 1;
    }
    int CaseIndex = 0;
    int NActive = NInds;
    while(NActive > 1 && CaseIndex < PerfSize)
    {
        int CurCase = OrderCases[CaseIndex];
        double tempBest = Perf[0][CurCase];
        for(int i=0;i!=NInds;i++)
        {
            if(Perf[i][CurCase] < tempBest)
                tempBest = Perf[i][CurCase];
        }
        double Med = MedAbsDist[CurCase];
        for(int i=0;i!=NInds;i++)
        {
            if(fabs(tempBest - Perf[i][CurCase]) > Med)
            {
                NActive -= IndActive[i];
                IndActive[i] = 0;
            }
            if(NActive == 1)
                break;
        }
        CaseIndex++;
    }
    int Selected = IntRandom(NActive);
    for(int i=0;i!=NInds;i++)
    {
        if(Selected == 0 && IndActive[i] == 1)
        {
            Selected = i;
            break;
        }
        Selected -= IndActive[i];
    }
    return Selected;
}
int Forest::LexicaseSelectionDS()
{
    for(int i=0;i!=PerfSize*5;i++)
    {
        swap(OrderCases[IntRandom(PerfSize)],OrderCases[IntRandom(PerfSize)]);
    }
    for(int i=0;i!=NInds;i++)
    {
        IndActive[i] = 1;
    }
    int CaseIndex = 0;
    int CaseNumber = 0;
    int NActive = NInds;
    while(NActive > 1 && CaseNumber < PerfSize && CaseIndex < PerfSize)
    {
        int CurCase = OrderCases[CaseIndex];
        if(InstanceActive[CurCase] != 1)
        {
            CaseIndex++;
            if(CaseIndex >= PerfSize)
                break;
            continue;
        }
        double tempBest = Perf[0][CurCase];
        for(int i=0;i!=NInds;i++)
        {
            if(Perf[i][CurCase] < tempBest)
                tempBest = Perf[i][CurCase];
        }
        double Med = MedAbsDist[CurCase];
        for(int i=0;i!=NInds;i++)
        {
            if(fabs(tempBest - Perf[i][CurCase]) > Med)
            {
                NActive -= IndActive[i];
                IndActive[i] = 0;
            }
            if(NActive == 1)
                break;
        }
        CaseIndex++;
        CaseNumber++;
    }
    int Selected = IntRandom(NActive);
    for(int i=0;i!=NInds;i++)
    {
        if(Selected == 0 && IndActive[i] == 1)
        {
            Selected = i;
            break;
        }
        Selected -= IndActive[i];
    }
    return Selected;
}

void Forest::StandardCrossover(const int Num, const int Selected)
{
    int ResultingLength;
    for(int index = 0;index != 25; index++)
    {
        delete Popul[NInds+Num];
        Popul[NInds+Num] = new node(1,NVars,0,0,0);
        Popul[Num]->copy_tree(Popul[NInds+Num]);

        int RandomNode1 = IntRandom(Popul[NInds+Num]->get_num());
        int RandomNode2 = IntRandom(Popul[Selected]->get_num());

        node *CP1, *CP2;
        Popul[NInds+Num]->get_crossPoint(RandomNode1, CP1);
        Popul[Selected]->get_crossPoint(RandomNode2, CP2);
        ResultingLength = Popul[NInds+Num]->get_num();
        //delete CP1;
        if(CP1->type == 2)
        {
            if(CP1->nar == 3)
            {
                delete CP1->next[2];
                delete CP1->next[1];
            }
            if(CP1->nar == 2)
                delete CP1->next[1];
            delete CP1->next[0];
            delete CP1->next;
        }
        //CP1 = new node(1,NVars,0,0,0);
        CP2->copy_tree(CP1);
        Popul[NInds+Num]->set_num(0);
        ResultingLength = Popul[NInds+Num]->get_num();
        //cout<<ResultingLength<<endl;
        if(ResultingLength < max_length)
            break;
    }

}

void Forest::Mutation(const int Num)
{
    int RandomNode = IntRandom(Popul[NInds+Num]->get_num());
    Popul[NInds+Num]->mutate(RandomNode,0.1,NVars,MaxDepth);
    Popul[NInds+Num]->set_num(0);
}

void Forest::PrintPoints(node* ind, sample &Samp)
{
    ofstream fout_sample("samp.txt");
    for(int i=0;i!=Samp.Size;i++)
    {
        if(problemtype == 0)
        {
            double selected = ind->operation(Samp.NormInputs[i]);
            fout_sample<<Samp.Classes[i]<<"\t"<<selected<<endl;
        }
        else
        {
            double selected = ind->operation(Samp.Inputs[i]);
            fout_sample<<Samp.Outputs[i][0]<<"\t"<<selected<<endl;
        }
    }
    ofstream fout_equat("equation.txt");
    fout_equat<<ind->print(buffer)<<endl;
}

void Forest::MainLoop(sample &Samp)
{
    if(SelType == 3 || SelType == 4 || ReplType == 5)
    {
        GenerateDS();
    }
    for(int i=0;i!=NInds;i++)
    {
        FitCalc(Popul[i],Samp,i);
    }
    //FindNSaveBest();
    //cout<<"best = "<<bestFit<<endl;
    do
    {
        double minfit = FitMass[0];
        double maxfit = FitMass[0];
        for(int i=0;i!=NInds;i++)
        {
            FitMassCopy[i] = FitMass[i];
            Indexes[i] = i;
            if(FitMass[i] >= maxfit)
                maxfit = FitMass[i];
            if(FitMass[i] <= minfit)
                minfit = FitMass[i];
        }
        if(minfit != maxfit)
            qSort2int(FitMassCopy,Indexes,0,NInds-1);
        FitTemp_vec.resize(NInds);
        for(int i=0;i!=NInds;i++)
            FitTemp_vec[i] = exp(double(i+1)/double(NInds)*3);
        std::discrete_distribution<int> ComponentSelector(FitTemp_vec.begin(),FitTemp_vec.end());
        //cout<<endl;
        //cout<<"sdfs"<<endl;
        if(SelType == 2)
        {
            PrepareLexicase(1);
        }
        if(SelType == 3)
        {
            GenerateDS();
            PrepareLexicaseDS(1);
//            for(int i=0;i!=NInds;i++)
//            {
//                FitMass[i] = FitCalc(Popul[i],Samp,i) + Popul[i]->get_num()*LenWeight;
//            }
//            FindNSaveBest();
        }
        if(SelType == 4)
        {
            PerfCalcFull(Popul[bestNum],Samp,bestNum);
            GenerateIS();
            PrepareLexicaseDS(1);
        }
        if(SelType == 5)
        {
            PerfCalcFull(Popul[bestNum],Samp,bestNum);
            GenerateIDS();
            PrepareLexicaseDS(1);
        }
        if(SelType != 3 && ReplType == 5)
        {
            GenerateDS();
        }
        int chosen=0;
        int chosen1 = 0;
        for(int i=0;i!=NInds;i++)
        {
            if(P1Type == 0)
                chosen1 = i;
            if(SelType == 0)
            {
                chosen = ComponentSelector(generator_uni_i_2);
                if(P1Type == 1)
                    chosen1 = ComponentSelector(generator_uni_i_2);
            }
            if(SelType == 1)
            {
                int besti = IntRandom(NInds);
                double bestf = FitMass[besti];
                for(int t=1;t!=Tsize;t++)
                {
                    int tempi = IntRandom(NInds);
                    if(FitMass[tempi] < bestf)
                    {
                        bestf = FitMass[tempi];
                        besti = tempi;
                    }
                }
                chosen = besti;
                if(P1Type == 1)
                {
                    besti = IntRandom(NInds);
                    bestf = FitMass[besti];
                    for(int t=1;t!=Tsize;t++)
                    {
                        int tempi = IntRandom(NInds);
                        if(FitMass[tempi] < bestf)
                        {
                            bestf = FitMass[tempi];
                            besti = tempi;
                        }
                    }
                    chosen1 = besti;
                }
            }
            if(SelType == 2)
            {
                chosen = LexicaseSelection();
                if(P1Type == 1)
                    chosen1 = LexicaseSelection();
            }
            if(SelType == 3)
            {
                chosen = LexicaseSelectionDS();
                if(P1Type == 1)
                    chosen1 = LexicaseSelectionDS();
            }
            if(SelType == 4)
            {
                chosen = LexicaseSelectionDS();
                if(P1Type == 1)
                    chosen1 = LexicaseSelectionDS();
            }
            if(SelType == 5)
            {
                chosen = LexicaseSelectionDS();
                if(P1Type == 1)
                    chosen1 = LexicaseSelectionDS();
            }
            StandardCrossover(chosen1, chosen);
            Mutation(i);
            FitCalc(Popul[NInds+i],Samp,NInds+i);
        }
        if(ReplType == 0)
        {
            CreateNewPop();
        }
        if(ReplType == 1)
        {
            CreateNewPopSort();
        }
        if(ReplType == 2)
        {
            CreateNewPopTour();
        }
        if(ReplType == 3)
        {
            CreateNewPopRank();
        }
        if(ReplType == 4)
        {
            PrepareLexicase(2);
            CreateNewPopLexicase();
        }
        if(ReplType == 5)
        {
            PrepareLexicaseDS(2);
            CreateNewPopLexicase();
        }
        if(ReplType == 6)
        {
            CreateNewPopOffspring();
        }
        cout<<"NFEvals\t"<<NFEvals<<"\tbest = "<<bestFit<<endl;
    }
    while(NFEvals < MaxFEvals);
    //cout<<"best = "<<bestFit<<endl;
}

Forest::~Forest()
{
    for(int i=0;i!=NInds*2;i++)
    {
        delete Popul[i];
        delete PopulTemp[i];
        delete Perf[i];
        delete PerfTemp[i];
    }
    delete Popul;
    delete PopulTemp;
    delete FitMass;
    delete FitTemp;
    delete Perf;
    delete Indexes;
    delete FitMassCopy;
    delete BestInd;
    delete SortPerf;
    delete MedAbsDist;
    delete MedAbsDistEps;
    delete MedAbsDistCopy;
    delete OrderCases;
    delete IndActive;
    delete InstanceActive;
    delete InstanceWeight;
    delete InstanceIndex;
    delete InstanceProb;
    delete InstanceRecCounter;
    delete NFESteps;
    delete AlreadyChosen;
}

int main()
{
    unsigned t0g=clock(),t1g;
    std::vector<std::string> datasets_path = {
    "datasets/192_vineyard.tsv",
    "datasets/210_cloud.tsv",
    "datasets/522_pm10.tsv",
    "datasets/557_analcatdata_apnea1.tsv",
    "datasets/579_fri_c0_250_5.tsv",
    "datasets/606_fri_c2_1000_10.tsv",
    "datasets/650_fri_c0_500_50.tsv",
    "datasets/678_visualizing_environmental.tsv",
    "datasets/1028_SWD.tsv",
    "datasets/1089_USCrime.tsv",
    "datasets/1193_BNG_lowbwt.tsv",
    "datasets/1199_BNG_echoMonths.tsv"
    };
    std::vector<int> dataset_size = {
    192,
    210,
    500,
    475,
    250,
    1000,
    500,
    111,
    1000,
    47,
    31104,
    17496};
    std::vector<int> dataset_vars = {
    2,
    5,
    7,
    3,
    5,
    10,
    50,
    3,
    10,
    13,
    9,
    9};
    int world_size=1,world_rank=0,name_len,TotalNRuns = 25;
    //MPI_Init(NULL, NULL);
    //MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    //MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    //char processor_name[MPI_MAX_PROCESSOR_NAME];
    //MPI_Get_processor_name(processor_name, &name_len);
    /*if(world_rank > TotalNRuns)
    {
        MPI_Finalize();
        return 0;
    }*/
    world_rank_global = world_rank;
    unsigned globalseed = world_rank;
    seed1 = mix(clock(),time(NULL),globalseed+getpid());
    seed2 = mix(clock(),time(NULL)+100,globalseed+getpid());
    seed3 = mix(clock(),time(NULL)+200,globalseed+getpid());
    seed4 = mix(clock(),time(NULL)+300,globalseed+getpid());
    seed4 = mix(clock(),time(NULL)+400,globalseed+getpid());
    generator_uni_i.seed(seed1);
    generator_uni_r.seed(seed2);
    generator_norm.seed(seed3);
    generator_cachy.seed(seed4);
    generator_uni_i_2.seed(seed5);


    int NTasks = 0;
    vector<Params> PRS;
    vector<int> TaskFinished;
    vector<Result> AllResults;
    vector<string> FilePath;
    int ResultSize = sizeof(Result);
    int* NodeBusy = new int[world_size-1];
    int* NodeTask = new int[world_size-1];
    for(int i=0;i!=world_size-1;i++)
    {
        NodeBusy[i] = 0;
        NodeTask[i] = -1;
    }
    Params tempPRS;
    Result tempResult;
    string tempFilePath;
    int TaskCounter = 0;
    int DiffPRSCounter = 0;
    string file_to_read = "";
    int SampleSizeTemp = 0;
    int NVars = 0;
    int MaxDepth = 5;
    int NInds = 100;
    int NGens = 0;
    int MaxFEvals = 100000;
    int CrossType = 0;
    int MutType = 0;
    int Tsize = 3;
    int max_length = 75;
    double InitProb = 0.1;
    double DSFactor = 5.0;
    double IncFactor = 0.1;
    //for(int run=0;run!=25;run++)
    for(int SelIter=3;SelIter!=4;SelIter++)    {
    for(int P1Iter=1;P1Iter!=2;P1Iter++)    {
    for(int ReplIter=0;ReplIter!=1;ReplIter++)    {
    for(int TsizeReplIter=3;TsizeReplIter!=4;TsizeReplIter++)    {

    DiffPRSCounter++;
    for (int RunN = 0;RunN!=TotalNRuns;RunN++)    {
        tempPRS.TaskN = TaskCounter;
        tempPRS.SelType = SelIter;
        tempPRS.ReplType = ReplIter;
        tempPRS.P1Type = P1Iter;
        tempPRS.TsizeRepl = TsizeReplIter;

        PRS.push_back(tempPRS);
        AllResults.push_back(tempResult);
        TaskFinished.push_back(0);

        string folder;
        folder = "GP18_Results";
        struct stat st = {0};
        if(world_rank == 0)
        {
            if (stat(folder.c_str(), &st) == -1)
                mkdir(folder.c_str(), 0777);
        }
        sprintf(buffer, "/Results_S%d_C%d_M%d_R%d_P%d_T%d_TR%d_MD%d_ML%d.txt",
            tempPRS.SelType,CrossType,MutType,tempPRS.ReplType,tempPRS.P1Type,Tsize,tempPRS.TsizeRepl,MaxDepth,max_length);
        tempFilePath = folder + buffer;
        FilePath.push_back(tempFilePath);
        TaskCounter++;
    }}}}}

    NTasks = TaskCounter;
    cout<<"Total\t"<<NTasks<<"\ttasks"<<endl;
    int PRSsize = sizeof(tempPRS);
    vector<int> ResSavedPerPRS;
    ResSavedPerPRS.resize(DiffPRSCounter);
    cout<<"ResSavedPerPRS"<<endl;
    for(int i=0;i!=DiffPRSCounter;i++)
    {
        ResSavedPerPRS[i] = 0;
        cout<<ResSavedPerPRS[i]<<"\t";
    }
    cout<<endl;
    if(world_rank > 0)
        sleep(0.01);

    /*if(world_rank == 0 && world_size > 1)
    {
        cout<<"Rank "<<world_rank<<" starting!"<<endl;
        int NFreeNodes = getNFreeNodes(NodeBusy,world_size-1);
        int NStartedTasks = getNStartedTasks(TaskFinished,NTasks);
        int NFinishedTasks = getNFinishedTasks(TaskFinished,NTasks);
        while((NStartedTasks < NTasks || NFinishedTasks < NTasks)  && world_size > 1)
        {
            NFreeNodes = getNFreeNodes(NodeBusy,world_size-1);
            int TaskToStart = -1;
            for(int i=0;i!=NTasks;i++)
            {
                if(TaskFinished[i] == 0)
                {
                    TaskToStart = i;
                    break;
                }
            }
            if(NFreeNodes > 0 && TaskToStart != -1)
            {
                int NodeToUse = -1;
                for(int i=0;i!=world_size-1;i++)
                {
                    if(NodeBusy[i] == 0)
                    {
                        NodeToUse = i;
                        break;
                    }
                }
                NodeTask[NodeToUse] = TaskToStart;
                TaskFinished[TaskToStart] = 1;
                //MPI_Send(&PRS[TaskToStart],PRSsize,MPI_BYTE,NodeToUse+1,0,MPI_COMM_WORLD);
                cout<<"sent task "<<TaskToStart<<" to "<<NodeToUse+1<<endl;
                NodeBusy[NodeToUse] = 1;
            }
            else
            {
                cout<<world_rank<<" Receiving result"<<endl;
                Result ReceivedRes;
                //MPI_Recv(&ReceivedRes,ResultSize,MPI_BYTE,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                AllResults[NodeTask[ReceivedRes.Node]].Copy(ReceivedRes,ResTsize1,ResTsize2);
                cout<<world_rank<<" received from \t"<<ReceivedRes.Node<<endl;
                NodeBusy[ReceivedRes.Node] = 0;
                TaskFinished[NodeTask[ReceivedRes.Node]] = 2;

                for(int i=0;i!=DiffPRSCounter;i++)
                {
                    int totalFinval = 0;
                    for(int j=0;j!=TotalNRuns;j++)
                    {
                        totalFinval += TaskFinished[i*TotalNRuns+j];
                    }
                    if(totalFinval == TotalNRuns*2 && ResSavedPerPRS[i] == 0)
                    {
                        cout<<"Saving to:  "<<FilePath[NodeTask[ReceivedRes.Node]]<<endl;
                        ResSavedPerPRS[i] = 1;
                        ofstream fout(FilePath[NodeTask[ReceivedRes.Node]]);
                        for(int RunN=0;RunN!=TotalNRuns;RunN++)
                        {
                            for(int func_num=0;func_num!=ResTsize1;func_num++)
                            {
                                for(int k=0;k!=ResTsize2;k++)
                                {
                                    fout<<AllResults[i*TotalNRuns+RunN].ResultTable1[func_num][k]<<"\t";
                                }
                                fout<<endl;
                            }
                        }
                        fout.close();
                    }
                }
            }
            string task_stat;
            task_stat = "";
            for(int i=0;i!=world_size-1;i++)
            {
                sprintf(buffer,"%d",NodeBusy[i]);
                task_stat += buffer;
            }
            cout<<"NodeBusy: "<<task_stat<<endl;
            cout<<"Total NTasks: "<<NTasks<<endl;

            task_stat = "";
            for(int i=0;i!=world_size-1;i++)
            {
                sprintf(buffer,"%d ",NodeTask[i]);
                task_stat += buffer;
            }
            cout<<"NodeTask: "<<task_stat<<endl;

            task_stat = "";
            for(int i=0;i!=NTasks;i++)
            {
                sprintf(buffer,"%d",TaskFinished[i]);
                task_stat += buffer;
            }
            NStartedTasks = getNStartedTasks(TaskFinished,NTasks);
            NFinishedTasks = getNFinishedTasks(TaskFinished,NTasks);
            cout<<"NFINISHED "<<NFinishedTasks<<endl;
        }
        cout<<world_rank<<" sending Finish"<<endl;
        for(int i=1;i!=world_size;i++)
        {
            Params PRSFinish;
            PRSFinish.Type = -1;
            //MPI_Send(&PRSFinish,PRSsize,MPI_BYTE,i,0,MPI_COMM_WORLD);
        }
    }
    else*/
    {
        int CurTask = 0;
        while(true)
        {
            Params CurPRS;
            if(world_size > 1)
            {
                //MPI_Recv(&CurPRS,PRSsize,MPI_BYTE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                cout<<world_rank<<" received"<<endl;
                if(CurPRS.Type == -1)
                {
                    cout<<world_rank<<" Finishing!"<<endl;
                    break;
                }
            }
            else
            {
                if(CurTask > NTasks)
                    break;
                CurPRS = PRS[CurTask];
            }
            Result ResToSend;
            ResToSend.Node=world_rank-1;
            ResToSend.Task=0;

            //////////////////////////////////////////////////////////////////////

            for(int func_num = 0; func_num < ResTsize1; func_num++)
            {
                //cout<<CurTask<<"\t"<<func_num<<endl;
                SampleSizeTemp = dataset_size[func_num];
                NVars = dataset_vars[func_num];
                file_to_read = datasets_path[func_num];
                Samp.Init(SampleSizeTemp,NVars,1,2,0.9,1);
                Samp.ReadFileRegression_SRBENCH((char*)file_to_read.c_str());
                Samp.SplitRandom();
                NVars = Samp.NVars;
                Samp.NormalizeCV_01(1);
                Forest NewForest(NVars,
                                 MaxDepth,
                                 NInds,
                                 NGens,
                                 MaxFEvals,
                                 CurPRS.SelType,
                                 CrossType,
                                 MutType,
                                 InitProb,
                                 Samp,
                                 1,
                                 CurPRS.ReplType,
                                 Tsize,
                                 DSFactor,
                                 IncFactor,
                                 CurPRS.P1Type,
                                 max_length,
                                 CurPRS.TsizeRepl);
                NewForest.MainLoop(Samp);
                for(int j=0;j!=ResTsize2;j++)
                {
                    ResToSend.ResultTable1[func_num][j] = ResultsArray[j];
                }
            }
            if(world_size > 1)
            {
                cout<<world_rank<<" sending to 0 result "<<endl;
                //MPI_Send(&ResToSend,ResultSize,MPI_BYTE,0,0,MPI_COMM_WORLD);
                sleep(0.1);
            }
            else
            {
                TaskFinished[CurTask] = 2;
                for(int i=0;i!=DiffPRSCounter;i++)
                {
                    int totalFinval = 0;
                    for(int j=0;j!=TotalNRuns;j++)
                    {
                        totalFinval += TaskFinished[i*TotalNRuns+j];
                    }
                    cout<<"totalFinVal "<<totalFinval<<endl;
                    if(totalFinval == TotalNRuns*2 && ResSavedPerPRS[i] == 0)
                    {
                        cout<<"Saving to:  "<<FilePath[CurTask]<<endl;
                        ResSavedPerPRS[i] = 1;
                        ofstream fout(FilePath[CurTask]);
                        for(int func_num=0;func_num!=ResTsize1;func_num++)
                        {
                            for(int k=0;k!=ResTsize2;k++)
                            {
                                fout<<AllResults[i*TotalNRuns+0].ResultTable1[func_num][k]<<"\t";
                            }
                            fout<<endl;
                        }
                        fout.close();
                    }
                }
                CurTask++;
            }
        }
    }

    delete[] NodeBusy;
    delete[] NodeTask;
    //if(world_rank == 0)
      //  cout<<"Rank "<<world_rank<<"\ton\t"<<processor_name<<"\t Finished at"<<"\t"<<currentDateTime()<<"\n";
    //MPI_Finalize();

    t1g=clock()-t0g;
    double T0g = t1g/double(CLOCKS_PER_SEC);
    if(world_rank == 0)
    {
        cout << "Time spent: " << T0g << endl;
    }
    return 0;
}
