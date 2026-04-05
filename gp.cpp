#include "sample.cpp"
//#include <mpi.h> //MPI disabled
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

const int n_oper = 25;
const double leftbordercosnt = -1.0;
const double rightbordercosnt = 1.0;
const double sigmaconst = 5.0;
const int problemtype = 1;  // 0 - classification, 1 - regression
int world_rank_global = 0;
char buffer[500];
sample Samp;

const int ResTsize1 = 96;
const int ResTsize2 = 150;

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
    int max_length;
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
    1,  /// x / y (AQ)
    0,  /// x ^ y
    1,  /// ln(1+x)
    0,  /// exp(x)
    1,  /// sqrt(x)
    1,  /// sin(x)
    1,  /// cos(x)
    0,  /// tan(x)
    0,  /// tanh(x)
    0,  /// 1.0/(x)
    0,  /// sinh(x)
    0,  /// cosh(x)
    1,  /// abs(x)
    1,  /// (x)^2
    1,  /// min(x,y)
    1   /// max(x,y)
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
        case 9: /// x / y (AQ)
        {return 2;}
        case 10: /// x ^ y
        {return 2;}
        case 11: /// ln(1+x)
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
        case 22: /// (x)^2
        {return 1;}
        case 23: /// min(x,y)
        {return 2;}
        case 24: /// max(x,y)
        {return 2;}
    }
    return 1;
}

class node
{
    public:
    node();
    node(const double new_nextprob, const double new_nx,
         const int new_depth, const int new_number, const int max_depth, const int max_length);
    ~node();
    double operation(double* xvals);
    int get_num();
    void set_num(const int new_number);
    void copy_tree(node* root);
    string print(char* buffer);
    void mutate(const int mutatednode, const double new_nextprob,
                const double new_nx, const int max_depth, const int max_length);
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
                  const double new_nx, const int max_depth, const int max_length)
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
                next[0] = new node(new_nextprob, new_nx, depth+1, number+1, max_depth, max_length>>1);
            }
            else if(nar == 2)
            {
                next = new node*[2];
                next[0] = new node(new_nextprob, new_nx, depth+1, number+1, max_depth, max_length>>1);
                int next_num = next[0]->get_num();
                next[1] = new node(new_nextprob, new_nx, depth+1, next_num+1, max_depth, max_length>>1);
            }
            else
            {
                next = new node*[3];
                next[0] = new node(new_nextprob, new_nx, depth+1, number+1, max_depth, max_length>>1);
                int next_num = next[0]->get_num();
                next[1] = new node(new_nextprob, new_nx, depth+1, next_num+1, max_depth, max_length>>1);
                next_num = next[1]->get_num();
                next[2] = new node(new_nextprob, new_nx, depth+1, next_num+1, max_depth, max_length>>1);
            }
        }
    }
    else
    {
        if(type == 2)
        {
            if(nar == 1)
                next[0]->mutate(mutatednode, new_nextprob, new_nx, max_depth, max_length>>1);
            else if(nar == 2)
            {
                next[0]->mutate(mutatednode, new_nextprob, new_nx, max_depth, max_length>>1);
                next[1]->mutate(mutatednode, new_nextprob, new_nx, max_depth, max_length>>1);
            }
            else
            {
                next[0]->mutate(mutatednode, new_nextprob, new_nx, max_depth, max_length>>1);
                next[1]->mutate(mutatednode, new_nextprob, new_nx, max_depth, max_length>>1);
                next[2]->mutate(mutatednode, new_nextprob, new_nx, max_depth, max_length>>1);
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
           const int new_depth, const int new_number, const int max_depth, const int max_length)
{
    //constant = Random(leftbordercosnt,rightbordercosnt);
    constant = NormRand(0,sigmaconst);
    xnum = IntRandom(new_nx);
    depth = new_depth;
    number = new_number;
    if(depth < max_depth && new_nextprob < Random(0,1) && number < max_length)
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
            next[0] = new node(new_nextprob, new_nx, depth+1, number+1, max_depth, max_length);
        }
        else if(nar == 2)
        {
            next = new node*[2];
            next[0] = new node(new_nextprob, new_nx, depth+1, number+1, max_depth, max_length);
            int next_num = next[0]->get_num();
            next[1] = new node(new_nextprob, new_nx, depth+1, next_num+1, max_depth, max_length);
        }
        else
        {
            next = new node*[3];
            next[0] = new node(new_nextprob, new_nx, depth+1, number+1, max_depth, max_length);
            int next_num = next[0]->get_num();
            next[1] = new node(new_nextprob, new_nx, depth+1, next_num+1, max_depth, max_length);
            next_num = next[1]->get_num();
            next[2] = new node(new_nextprob, new_nx, depth+1, next_num+1, max_depth, max_length);
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
            res = next[0]->operation(xvals) / sqrt(1.0+next[1]->operation(xvals)*next[1]->operation(xvals));
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
        case 11: /// ln(1+x)
        {
            res = log(1.0+next[0]->operation(xvals));
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
        case 22: /// (x)^2
        {
            res = (next[0]->operation(xvals))*(next[0]->operation(xvals));
            if(isinf(res) && res > 0)
                res = std::numeric_limits<double>::max();
            if(isinf(res) && res < 0)
                res = -std::numeric_limits<double>::max();
            if(isnan(res))
                res = 0;
            return res;
        }
        case 23: /// min(x,y)
        {
            res = min(next[0]->operation(xvals),next[1]->operation(xvals));
            if(isinf(res) && res > 0)
                res = std::numeric_limits<double>::max();
            if(isinf(res) && res < 0)
                res = -std::numeric_limits<double>::max();
            if(isnan(res))
                res = 0;
            return res;
        }
        case 24: /// max(x,y)
        {
            res = max(next[0]->operation(xvals),next[1]->operation(xvals));
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
        case 9: /// x / y (AQ)
        {return "(" + next[0]->print(buffer) + "/(1+(" + next[1]->print(buffer) + ")^2)";}
        case 10: /// x ^ y
        {return "(" + next[0]->print(buffer) + "^" + next[1]->print(buffer) + ")";}
        case 11: /// ln(1+x)
        {return "log(1+" + next[0]->print(buffer) + ")";}
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
        case 22: /// (x)^2
        {return "((" + next[0]->print(buffer) + ")^2)";}
        case 23: /// abs(x)
        {return "min(" + next[0]->print(buffer) + "," + next[1]->print(buffer) + ")";}
        case 24: /// abs(x)
        {return "max(" + next[0]->print(buffer) + "," + next[1]->print(buffer) + ")";}
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
    node* BestInd;
    node* TempInd;
    int* Indexes;
    int* OrderCases;
    int* IndActive;
    int* InstanceActive;
    int* InstanceIndex;
    int* NFESteps;
    int* AlreadyChosen;
    int* CaseRanks;
    int* GFR_indexes;
    int** IndexPerf;
    double* FitMass;
    double** Perf;
    double* SortPerf;
    double* FitMassCopy;
    double* MedAbsDist;
    double* MedAbsDistEps;
    double* MedAbsDistEpsFront;
    double* GFR_TempVals;
    double* GFR_TempRanks;
    double* Ranks;
    double* GFR_TVals;
    double* GFR_TRanks;
    vector<double> FitTemp_vec;
    vector<double> InstProb;
    double LenWeight = 1e-8;
    int PerfSize;
    int PerfSizeTrain;
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
    int FrontOffset;
    double IncFactor;
    double InitProb;
    double bestFit;
    double bestTestFit;
    double DSFactor;
    void FitCalc(node* ind, sample &Samp, int index);
    void PerfCalcFull(node* ind, sample &Samp, int index);
    double FitCalcTest(node* ind, sample &Samp);
    void PrintPoints(node* ind, sample &Samp);
    void MainLoop(sample &Samp);
    void FindNSaveBest();
    void PrepareLexicaseDS(int SizeMultiplier);
    void PrepareFriedmanSel(int SizeMultiplier);
    void GenerateDS();
    void CreateNewPopOffspring();
    void CreateNewPopPairWise();
    void CreateNewPopSort();
    void CreateNewPopFriedman();
    int TourSelection(int CurOffset);
    int LexicaseSelectionDS(int CurOffset);
    int LexicaseSelectionCS(int chosen, int CurOffset);
    int FriedmanSelection(int CurOffset);
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
    FrontOffset = NInds*2;
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
    GFR_indexes = new int[NInds*2];
    MedAbsDist = new double[PerfSize];
    MedAbsDistEps = new double[PerfSize];
    MedAbsDistEpsFront = new double[PerfSize];
    CaseRanks = new int[PerfSize];
    IndexPerf = new int*[PerfSize];
    SortPerf = new double[NInds*3];
    GFR_TempVals = new double[NInds*2];
    GFR_TempRanks = new double[NInds*2];
    Ranks = new double[NInds*3];
    GFR_TVals = new double[NInds*2];
    GFR_TRanks = new double[NInds*2];
    FitMass = new double[NInds*3];
    Perf = new double*[NInds*3];
    Indexes = new int[NInds*2];
    FitMassCopy = new double[NInds*2];
    FitTemp_vec.resize(NInds);
    InstProb.resize(PerfSize);
    Popul = new node*[NInds*3];
    for(int i=0;i!=PerfSize;i++)
    {
        IndexPerf[i] = new int[NInds*3];
    }
    BestInd = new node(new_InitProb,NVars,0,0,MaxDepth,max_length);
    TempInd = new node(new_InitProb,NVars,0,0,MaxDepth,max_length);
    for(int i=0;i!=NInds;i++)
    {
        Popul[i] = new node(new_InitProb,NVars,0,0,MaxDepth,max_length);
        Popul[NInds+i] = new node(1,NVars,0,0,0,0);
        Popul[FrontOffset+i] = new node(1,NVars,0,0,0,0);
        Popul[i]->copy_tree(Popul[FrontOffset+i]);
        Perf[i] = new double[Samp.Size];
        Perf[NInds+i] = new double[Samp.Size];
        Perf[FrontOffset+i] = new double[Samp.Size];
    }
    /*if(SelType == 0 || SelType == 1 || SelType == 2 || SelType == 6)*/
    if(P1Type == 0)
    {
        int casecounter = 0;
        for(int i=0;i!=PerfSize;i++)
        {
            if(Samp.GetCVFoldNum(i) != 1)
            {
                InstanceIndex[casecounter] = i;
                casecounter++;
            }
        }
        PerfSizeTrain = casecounter;
        for(int i=0;i!=PerfSize;i++)
        {
            if(Samp.GetCVFoldNum(i) == 1)
            {
                InstanceIndex[casecounter] = i;
                casecounter++;
            }
        }
        for(int i=0;i!=PerfSizeTrain;i++)
        {
            InstanceActive[InstanceIndex[i]] = 1;
        }
        for(int i=PerfSizeTrain;i!=PerfSize;i++)
        {
            InstanceActive[InstanceIndex[i]] = 0;
        }
        PerfSizeDS = PerfSizeTrain;
    }
    else
    /*if(SelType == 3 || SelType == 4 || SelType == 5)*/
    {
        int casecounter = 0;
        for(int i=0;i!=PerfSize;i++)
        {
            if(Samp.GetCVFoldNum(i) != 1)
            {
                casecounter+=1;
            }
        }
        PerfSizeTrain = casecounter;
        PerfSizeDS = int(double(PerfSizeTrain)/double(DSFactor));
    }
}

void Forest::FitCalc(node* ind, sample &Samp, int index)
{
    double Error = 0;
    double Error2 = 0;
    double Mean = 0;
    for(int i=0;i!=PerfSize;i++)
    {
        if(Samp.GetCVFoldNum(i) != 1)//InstanceActive[i] == 1) // &&
        {
            Mean += Samp.Outputs[i][0];
        }
    }
    Mean /= double(PerfSizeDS);
    for(int i=0;i!=PerfSize;i++)
    {
        if(Samp.GetCVFoldNum(i) != 1)//InstanceActive[i] == 1) // &&
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
        BestInd = new node(1,NVars,0,0,0,0);
        Popul[bestNum]->copy_tree(BestInd);
        bestTestFit = FitCalcTest(BestInd, Samp);
    }
    if(NFEvals == NFESteps[LastStep])
    {
        ResultsArray[LastStep*3+0] = -bestFit;
        ResultsArray[LastStep*3+1] = -bestTestFit;
        ResultsArray[LastStep*3+2] = BestInd->get_num();
        //cout<<NFEvals<<"\tbest = "<<bestFit<<"\t"<<bestTestFit<<endl;
        LastStep++;
    }
    //cout<<Error<<endl;
}
double Forest::FitCalcTest(node* ind, sample &Samp)
{
    double Error = 0;
    double Error2 = 0;
    double Mean = 0;
    for(int i=0;i!=Samp.Size;i++)
    {
        if(Samp.GetCVFoldNum(i) == 1)
        {
            Mean += Samp.Outputs[i][0];
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
void Forest::CreateNewPopOffspring()
{
    int worstindex = 0;
    double worstfit = FitMass[0];
    for(int i=0;i!=NInds;i++)
    {
        delete Popul[i];
        Popul[i] = new node(1,NVars,0,0,0,0);
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
    Popul[worstindex] = new node(1,NVars,0,0,0,0);
    BestInd->copy_tree(Popul[worstindex]);
}
void Forest::CreateNewPopPairWise()
{
    for(int i=0;i!=NInds;i++)
    {
        if(FitMass[NInds+i] < FitMass[i])
        {
            delete Popul[i];
            Popul[i] = new node(1,NVars,0,0,0,0);
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
    }
}
void Forest::CreateNewPopFriedman()
{
    for(int i=0;i!=NInds*2;i++)
        Ranks[i] = 0;
    for(int c=0;c!=PerfSize;c++)
    {
        for(int i=0;i!=NInds*2;i++)
            GFR_TempVals[i] = Perf[i][c];
        get_fract_ranks(GFR_TempVals, GFR_TempRanks, NInds*2, GFR_indexes, GFR_TVals, GFR_TRanks);
        for(int i=0;i!=NInds*2;i++)
            Ranks[i] += GFR_TempRanks[i];
    }
    for(int i=0;i!=NInds*2;i++)
    {
        for(int j=i;j!=NInds*2;j++)
        {
            if(Ranks[i] > Ranks[j]) //replace
            {
                swap(Ranks[i],Ranks[j]);
                swap(Popul[i],Popul[j]);
                swap(FitMass[i],FitMass[j]);
                swap(Perf[i],Perf[j]);
            }
        }
    }
    //for(int i=0;i!=NInds*2;i++)
    //    cout<<FitMass[i]<<"\t"<<Ranks[i]<<"\n";
    //cout<<endl;
}
void Forest::PrepareLexicaseDS(int CurrOffset)
{
    int casecounter = 0;
    for(int c=0;c!=PerfSize;c++)
    {
        if(InstanceActive[c] == 1)
        {
            for(int i=0;i!=NInds;i++)
            {
                SortPerf[i] = Perf[CurrOffset+i][c];
                IndexPerf[c][CurrOffset+i] = CurrOffset+i;
            }
            qSort2int(SortPerf,IndexPerf[c],0,NInds-1);
            MedAbsDist[c] = SortPerf[(NInds) >> 1];
            for(int i=0;i!=NInds;i++)
            {
                SortPerf[i] = fabs(Perf[CurrOffset+i][c]-MedAbsDist[c]);
            }
            qSort1(SortPerf,0,NInds-1);
            if(CurrOffset == 0)
                MedAbsDistEps[c] = SortPerf[(NInds) >> 1];
            else
                MedAbsDistEpsFront[c] = SortPerf[(NInds) >> 1];
            casecounter++;
        }
        OrderCases[c] = c;
    }
}
void Forest::GenerateDS()
{
    int casecounter = 0;
    for(int i=0;i!=PerfSize;i++)
    {
        if(Samp.GetCVFoldNum(i) != 1)
        {
            InstanceIndex[casecounter] = i;
            casecounter++;
        }
    }
    for(int i=0;i!=PerfSize;i++)
    {
        if(Samp.GetCVFoldNum(i) == 1)
        {
            InstanceIndex[casecounter] = i;
            casecounter++;
        }
    }
    for(int i=0;i!=PerfSizeTrain*10;i++)
    {
        swap(InstanceIndex[IntRandom(PerfSizeTrain)],InstanceIndex[IntRandom(PerfSizeTrain)]);
    }
    for(int i=0;i!=PerfSizeDS;i++)
    {
        InstanceActive[InstanceIndex[i]] = 1;
    }
    for(int i=PerfSizeDS;i!=PerfSize;i++)
    {
        InstanceActive[InstanceIndex[i]] = 0;
    }
}
void Forest::PrepareFriedmanSel(int CurrOffset)
{
    for(int i=0;i!=NInds;i++)
        Ranks[i] = 0;
    for(int c=0;c!=PerfSize;c++)
    {
        if(InstanceActive[c] == 1)
        {
            for(int i=0;i!=NInds;i++)
                GFR_TempVals[i] = Perf[i][c];
            get_fract_ranks(GFR_TempVals, GFR_TempRanks, NInds, GFR_indexes, GFR_TVals, GFR_TRanks);
            for(int i=0;i!=NInds;i++)
                Ranks[i] += GFR_TempRanks[i];
        }
    }

    for(int i=0;i!=NInds;i++)
        Ranks[CurrOffset+i] = 0;
    for(int c=0;c!=PerfSize;c++)
    {
        if(InstanceActive[c] == 1)
        {
            for(int i=0;i!=NInds;i++)
                GFR_TempVals[i] = Perf[CurrOffset+i][c];
            get_fract_ranks(GFR_TempVals, GFR_TempRanks, NInds, GFR_indexes, GFR_TVals, GFR_TRanks);
            for(int i=0;i!=NInds;i++)
                Ranks[CurrOffset+i] += GFR_TempRanks[i];
        }
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
    BestInd = new node(1,NVars,0,0,0,0);
    Popul[bestNum]->copy_tree(BestInd);
}

int Forest::TourSelection(int CurrOffset)
{
    int besti = IntRandom(NInds);
    double bestf = FitMass[CurrOffset+besti];
    for(int t=1;t!=Tsize;t++)
    {
        int tempi = IntRandom(NInds);
        if(FitMass[CurrOffset+tempi] < bestf)
        {
            bestf = FitMass[CurrOffset+tempi];
            besti = tempi;
        }
    }
    return besti;
}
int Forest::FriedmanSelection(int CurrOffset)
{
    int besti = IntRandom(NInds);
    double bestf = Ranks[CurrOffset+besti];
    for(int t=1;t!=Tsize;t++)
    {
        int tempi = IntRandom(NInds);
        if(Ranks[CurrOffset+tempi] < bestf)
        {
            bestf = Ranks[CurrOffset+tempi];
            besti = tempi;
        }
    }
    return besti;
}
int Forest::LexicaseSelectionDS(int CurOffset)
{
    for(int i=0;i!=PerfSizeTrain*5;i++)
    {
        swap(OrderCases[IntRandom(PerfSizeTrain)],OrderCases[IntRandom(PerfSizeTrain)]);
    }
    for(int i=0;i!=NInds;i++)
    {
        IndActive[i] = 1;
    }
    int CaseIndex = 0;
    int CaseNumber = 0;
    int NActive = NInds;
    while(NActive > 1 && CaseNumber < PerfSizeTrain && CaseIndex < PerfSizeTrain)
    {
        int CurCase = OrderCases[CaseIndex];
        if(InstanceActive[CurCase] != 1)
        {
            CaseIndex++;
            if(CaseIndex >= PerfSizeTrain)
                break;
            continue;
        }
        double tempBest = Perf[CurOffset+0][CurCase];
        for(int i=0;i!=NInds;i++)
        {
            if(Perf[CurOffset+i][CurCase] < tempBest)
                tempBest = Perf[CurOffset+i][CurCase];
        }
        for(int i=0;i!=NInds;i++)
        {
            double med = 0;
            if(CurOffset == 0)
                med = MedAbsDistEps[CurCase];
            else
                med = MedAbsDistEpsFront[CurCase];
            if(Perf[CurOffset+i][CurCase] > tempBest + med)
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
    return CurOffset+Selected;
}
int Forest::LexicaseSelectionCS(int chosen, int CurOffset)
{
    int casecounter = 0;
    for(int i=0;i!=PerfSize;i++)
    {
        if(InstanceActive[i] == 1)
        {
            for(int j=0;j!=NInds;j++)
            {
                if(IndexPerf[i][j] == chosen)
                {
                    CaseRanks[casecounter] = -j;
                    break;
                }
            }
            OrderCases[casecounter] = i;
            casecounter++;
        }
    }
    qSortintint(CaseRanks,OrderCases,0,PerfSizeDS-1);

    for(int i=0;i!=NInds;i++)
    {
        IndActive[i] = 1;
    }
    IndActive[chosen] = 0;
    int CaseIndex = 0;
    int CaseNumber = 0;
    int NActive = NInds;
    while(NActive > 1 && CaseNumber < PerfSizeDS && CaseIndex < PerfSizeDS)
    {
        int CurCase = OrderCases[CaseIndex];
        if(InstanceActive[CurCase] != 1)
        {
            CaseIndex++;
            if(CaseIndex >= PerfSizeDS)
                break;
            continue;
        }
        double tempBest = Perf[0][CurCase];
        for(int i=0;i!=NInds;i++)
        {
            if(Perf[i][CurCase] < tempBest)
                tempBest = Perf[i][CurCase];
        }
        for(int i=0;i!=NInds;i++)
        {
            if(Perf[i][CurCase] > tempBest + MedAbsDistEps[CurCase])
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
    int CrossoverMade = 0;
    delete TempInd;
    TempInd = new node(1,NVars,0,0,0,0);
    Popul[Num]->copy_tree(TempInd);
    for(int index = 0;index != 25; index++)
    {
        delete Popul[NInds+Num];
        Popul[NInds+Num] = new node(1,NVars,0,0,0,0);
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
        {
            CrossoverMade = 1;
            break;
        }
    }
    if(CrossoverMade == 0)
    {
        //cout<<"Crossover failure"<<endl;
        delete Popul[NInds+Num];
        Popul[NInds+Num] = new node(1,NVars,0,0,0,0);
        TempInd->copy_tree(Popul[NInds+Num]);
    }
}
void Forest::Mutation(const int Num)
{
    int ResultingLength;
    int MutationMade = 0;
    delete TempInd;
    TempInd = new node(1,NVars,0,0,0,0);
    Popul[NInds+Num]->copy_tree(TempInd);
    for(int index = 0;index != 25; index++)
    {
        int RandomNode = IntRandom(Popul[NInds+Num]->get_num());
        Popul[NInds+Num]->mutate(RandomNode,0.1,NVars,MaxDepth,max_length);
        Popul[NInds+Num]->set_num(0);
        ResultingLength = Popul[NInds+Num]->get_num();
        if(ResultingLength < max_length)
        {
            MutationMade = 1;
            break;
        }
    }
    if(MutationMade == 0)
    {
        //cout<<"Mutation failure"<<endl;
        delete Popul[NInds+Num];
        Popul[NInds+Num] = new node(1,NVars,0,0,0,0);
        TempInd->copy_tree(Popul[NInds+Num]);
    }
}

void Forest::PrintPoints(node* ind, sample &Samp)
{
    ofstream fout_sample("samp.tsv");
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
    ofstream fout_equat("equation.tsv");
    fout_equat<<ind->print(buffer)<<endl;
}

void Forest::MainLoop(sample &Samp)
{
    //if(SelType == 3 || SelType == 4 || ReplType == 6)
    if(P1Type == 1)
    {
        GenerateDS();
    }
    for(int i=0;i!=NInds;i++)
    {
        FitCalc(Popul[i],Samp,i);
        FitMass[FrontOffset+i] = FitMass[i];
        for(int j=0;j!=PerfSize;j++)
        {
            Perf[FrontOffset+i][j] = Perf[i][j];
        }
    }
    int PFIndex = 0;
    do
    {
        /*double minfit = FitMass[0];
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
        std::discrete_distribution<int> ComponentSelector(FitTemp_vec.begin(),FitTemp_vec.end());*/
        /*if(SelType == 2)
        {
            PrepareLexicaseDS(1);
        }
        if(SelType == 3 || SelType == 4 || SelType == 5)
        {
            GenerateDS();
            PrepareLexicaseDS(1);
        }
        if(SelType != 3 && ReplType == 6)
        {
            GenerateDS();
        }
        if(SelType == 6)
        {
            PrepareFriedmanSel(FrontOffset);
        }*/
        if(P1Type == 1)
            GenerateDS();
        PrepareLexicaseDS(1);
        PrepareFriedmanSel(0);
        PrepareFriedmanSel(FrontOffset);
        int chosen1 = 0;
        int chosen2 = 0;
        for(int i=0;i!=NInds;i++)
        {
            //if(P1Type == 0)
                //chosen1 = i;
            /*if(SelType == 0)
            {
                if(P1Type == 1)
                    chosen1 = ComponentSelector(generator_uni_i_2);
                chosen2 = ComponentSelector(generator_uni_i_2);
            }
            if(SelType == 2)
            {
                chosen1 = LexicaseSelectionDS(0);
                chosen2 = LexicaseSelectionCS(chosen1,0);
            }
            */

            /*
            i
            tour
            lex
            fried
            */
            if(SelType == 0)
            {
                chosen1 = i;
                chosen2 = i;
            }
            if(SelType == 1)
            {
                chosen1 = i;
                int tempcount = 0;
                do {
                chosen2 = TourSelection(0);
                tempcount++;
                if(tempcount == 5)
                {
                    chosen2 = (chosen1+1)%NInds;
                    break;
                }
                } while(chosen2 == chosen1);
            }
            if(SelType == 2)
            {
                chosen1 = i;
                int tempcount = 0;
                do {
                chosen2 = TourSelection(FrontOffset);
                tempcount++;
                if(tempcount == 5)
                {
                    chosen2 = (chosen1+1)%NInds;
                    break;
                }
                } while(chosen2 == chosen1);
            }
            if(SelType == 3)
            {
                chosen1 = i;
                int tempcount = 0;
                do {
                chosen2 = LexicaseSelectionDS(0);
                tempcount++;
                if(tempcount == 5)
                {
                    chosen2 = (chosen1+1)%NInds;
                    break;
                }
                } while(chosen2 == chosen1);
            }
            if(SelType == 4)
            {
                chosen1 = i;
                int tempcount = 0;
                do {
                chosen2 = LexicaseSelectionDS(FrontOffset);
                tempcount++;
                if(tempcount == 5)
                {
                    chosen2 = (chosen1+1)%NInds;
                    break;
                }
                } while(chosen2 == chosen1);
            }
            if(SelType == 5)
            {
                chosen1 = i;
                int tempcount = 0;
                do {
                chosen2 = FriedmanSelection(0);
                tempcount++;
                if(tempcount == 5)
                {
                    chosen2 = (chosen1+1)%NInds;
                    break;
                }
                } while(chosen2 == chosen1);
            }
            if(SelType == 6)
            {
                chosen1 = i;
                int tempcount = 0;
                do {
                chosen2 = FriedmanSelection(FrontOffset);
                tempcount++;
                if(tempcount == 5)
                {
                    chosen2 = (chosen1+1)%NInds;
                    break;
                }
                } while(chosen2 == chosen1);
            }
            if(SelType == 7)
            {
                chosen1 = TourSelection(0);
                int tempcount = 0;
                do {
                chosen2 = TourSelection(0);
                tempcount++;
                if(tempcount == 5)
                {
                    chosen2 = (chosen1+1)%NInds;
                    break;
                }
                } while(chosen2 == chosen1);
            }
            if(SelType == 8)
            {
                chosen1 = TourSelection(0);
                int tempcount = 0;
                do {
                chosen2 = TourSelection(FrontOffset);
                tempcount++;
                if(tempcount == 5)
                {
                    chosen2 = (chosen1+1)%NInds;
                    break;
                }
                } while(chosen2 == chosen1);
            }
            if(SelType == 9)
            {
                chosen1 = TourSelection(0);
                int tempcount = 0;
                do {
                chosen2 = LexicaseSelectionDS(0);
                tempcount++;
                if(tempcount == 5)
                {
                    chosen2 = (chosen1+1)%NInds;
                    break;
                }
                } while(chosen2 == chosen1);
            }
            if(SelType == 10)
            {
                chosen1 = TourSelection(0);
                int tempcount = 0;
                do {
                chosen2 = LexicaseSelectionDS(FrontOffset);
                tempcount++;
                if(tempcount == 5)
                {
                    chosen2 = (chosen1+1)%NInds;
                    break;
                }
                } while(chosen2 == chosen1);
            }
            if(SelType == 11)
            {
                chosen1 = TourSelection(0);
                int tempcount = 0;
                do {
                chosen2 = FriedmanSelection(0);
                tempcount++;
                if(tempcount == 5)
                {
                    chosen2 = (chosen1+1)%NInds;
                    break;
                }
                } while(chosen2 == chosen1);
            }
            if(SelType == 12)
            {
                chosen1 = TourSelection(0);
                int tempcount = 0;
                do {
                chosen2 = FriedmanSelection(FrontOffset);
                tempcount++;
                if(tempcount == 5)
                {
                    chosen2 = (chosen1+1)%NInds;
                    break;
                }
                } while(chosen2 == chosen1);
            }
            if(SelType == 13)
            {
                chosen1 = TourSelection(FrontOffset);
                int tempcount = 0;
                do {
                chosen2 = LexicaseSelectionDS(0);
                tempcount++;
                if(tempcount == 5)
                {
                    chosen2 = (chosen1+1)%NInds;
                    break;
                }
                } while(chosen2 == chosen1);
            }
            if(SelType == 14)
            {
                chosen1 = TourSelection(FrontOffset);
                int tempcount = 0;
                do {
                chosen2 = LexicaseSelectionDS(FrontOffset);
                tempcount++;
                if(tempcount == 5)
                {
                    chosen2 = (chosen1+1)%NInds;
                    break;
                }
                } while(chosen2 == chosen1);
            }
            if(SelType == 15)
            {
                chosen1 = TourSelection(FrontOffset);
                int tempcount = 0;
                do {
                chosen2 = FriedmanSelection(0);
                tempcount++;
                if(tempcount == 5)
                {
                    chosen2 = (chosen1+1)%NInds;
                    break;
                }
                } while(chosen2 == chosen1);
            }
            if(SelType == 16)
            {
                chosen1 = TourSelection(FrontOffset);
                int tempcount = 0;
                do {
                chosen2 = FriedmanSelection(FrontOffset);
                tempcount++;
                if(tempcount == 5)
                {
                    chosen2 = (chosen1+1)%NInds;
                    break;
                }
                } while(chosen2 == chosen1);
            }
            if(SelType == 17)
            {
                chosen1 = LexicaseSelectionDS(0);
                int tempcount = 0;
                do {
                chosen2 = LexicaseSelectionDS(0);
                tempcount++;
                if(tempcount == 5)
                {
                    chosen2 = (chosen1+1)%NInds;
                    break;
                }
                } while(chosen2 == chosen1);
            }
            if(SelType == 18)
            {
                chosen1 = LexicaseSelectionDS(0);
                int tempcount = 0;
                do {
                chosen2 = LexicaseSelectionDS(FrontOffset);
                tempcount++;
                if(tempcount == 5)
                {
                    chosen2 = (chosen1+1)%NInds;
                    break;
                }
                } while(chosen2 == chosen1);
            }
            if(SelType == 19)
            {
                chosen1 = LexicaseSelectionDS(0);
                int tempcount = 0;
                do {
                chosen2 = FriedmanSelection(0);
                tempcount++;
                if(tempcount == 5)
                {
                    chosen2 = (chosen1+1)%NInds;
                    break;
                }
                } while(chosen2 == chosen1);
            }
            if(SelType == 20)
            {
                chosen1 = LexicaseSelectionDS(0);
                int tempcount = 0;
                do {
                chosen2 = FriedmanSelection(FrontOffset);
                tempcount++;
                if(tempcount == 5)
                {
                    chosen2 = (chosen1+1)%NInds;
                    break;
                }
                } while(chosen2 == chosen1);
            }
            if(SelType == 21)
            {
                chosen1 = FriedmanSelection(0);
                int tempcount = 0;
                do {
                chosen2 = FriedmanSelection(0);
                tempcount++;
                if(tempcount == 5)
                {
                    chosen2 = (chosen1+1)%NInds;
                    break;
                }
                } while(chosen2 == chosen1);
            }
            if(SelType == 22)
            {
                chosen1 = FriedmanSelection(0);
                int tempcount = 0;
                do {
                chosen2 = FriedmanSelection(FrontOffset);
                tempcount++;
                if(tempcount == 5)
                {
                    chosen2 = (chosen1+1)%NInds;
                    break;
                }
                } while(chosen2 == chosen1);
            }
            if(SelType == 23)
            {
                chosen1 = FriedmanSelection(FrontOffset);
                int tempcount = 0;
                do {
                chosen2 = FriedmanSelection(FrontOffset);
                tempcount++;
                if(tempcount == 5)
                {
                    chosen2 = (chosen1+1)%NInds;
                    break;
                }
                } while(chosen2 == chosen1);
            }

            StandardCrossover(chosen1, chosen2);
            Mutation(i);
            FitCalc(Popul[NInds+i],Samp,NInds+i);
            if(FitMass[NInds+i] <= FitMass[i])
            {
                delete Popul[FrontOffset+PFIndex];
                Popul[FrontOffset+PFIndex] = new node(1,NVars,0,0,0,0);
                Popul[NInds+i]->copy_tree(Popul[FrontOffset+PFIndex]);
                FitMass[FrontOffset+PFIndex] = FitMass[NInds+i];
                for(int j=0;j!=PerfSize;j++)
                {
                    Perf[FrontOffset+PFIndex][j] = Perf[NInds+i][j];
                }
                PFIndex = (PFIndex+1)%NInds;
            }
        }
        if(ReplType == 0)
        {
            CreateNewPopOffspring();
        }
        if(ReplType == 1)
        {
            CreateNewPopPairWise();
        }
        if(ReplType == 2)
        {
            CreateNewPopSort();
        }
        if(ReplType == 3)
        {
            CreateNewPopFriedman();
        }
        //cout<<NFEvals<<"\tbest = "<<bestFit<<"\t"<<bestTestFit<<"\t"<<BestInd->print(buffer)<<endl;
    }
    while(NFEvals < MaxFEvals);
    //cout<<"best = "<<bestFit<<endl;
}

Forest::~Forest()
{
    for(int i=0;i!=PerfSize;i++)
    {
        delete IndexPerf[i];
    }
    for(int i=0;i!=NInds*3;i++)
    {
        delete Popul[i];
        delete Perf[i];
    }
    delete Popul;
    delete FitMass;
    delete Perf;
    delete Indexes;
    delete FitMassCopy;
    delete BestInd;
    delete SortPerf;
    delete MedAbsDist;
    delete MedAbsDistEps;
    delete MedAbsDistEpsFront;
    delete OrderCases;
    delete IndActive;
    delete InstanceActive;
    delete InstanceIndex;
    delete NFESteps;
    delete AlreadyChosen;
    delete CaseRanks;
    delete GFR_indexes;
    delete GFR_TempVals;
    delete GFR_TempRanks;
    delete Ranks;
    delete GFR_TVals;
    delete GFR_TRanks;
}

int main()
{
    unsigned t0g=clock(),t1g;
    std::vector<std::string> datasets_path = {
    "datasets/1027_ESL.tsv",
    "datasets/1028_SWD.tsv",
    "datasets/1029_LEV.tsv",
    "datasets/1030_ERA.tsv",
    "datasets/1089_USCrime.tsv",
    "datasets/1096_FacultySalaries.tsv",
    "datasets/192_vineyard.tsv",
    "datasets/210_cloud.tsv",
    "datasets/228_elusage.tsv",
    "datasets/229_pwLinear.tsv",
    "datasets/230_machine_cpu.tsv",
    "datasets/4544_GeographicalOriginalofMusic.tsv",
    "datasets/485_analcatdata_vehicle.tsv",
    "datasets/505_tecator.tsv",
    "datasets/519_vinnie.tsv",
    "datasets/522_pm10.tsv",
    "datasets/523_analcatdata_neavote.tsv",
    "datasets/527_analcatdata_election2000.tsv",
    "datasets/542_pollution.tsv",
    "datasets/547_no2.tsv",
    "datasets/556_analcatdata_apnea2.tsv",
    "datasets/557_analcatdata_apnea1.tsv",
    "datasets/560_bodyfat.tsv",
    "datasets/561_cpu.tsv",
    "datasets/579_fri_c0_250_5.tsv",
    "datasets/581_fri_c3_500_25.tsv",
    "datasets/582_fri_c1_500_25.tsv",
    "datasets/583_fri_c1_1000_50.tsv",
    "datasets/584_fri_c4_500_25.tsv",
    "datasets/586_fri_c3_1000_25.tsv",
    "datasets/588_fri_c4_1000_100.tsv",
    "datasets/589_fri_c2_1000_25.tsv",
    "datasets/590_fri_c0_1000_50.tsv",
    "datasets/591_fri_c1_100_10.tsv",
    "datasets/592_fri_c4_1000_25.tsv",
    "datasets/593_fri_c1_1000_10.tsv",
    "datasets/594_fri_c2_100_5.tsv",
    "datasets/595_fri_c0_1000_10.tsv",
    "datasets/596_fri_c2_250_5.tsv",
    "datasets/597_fri_c2_500_5.tsv",
    "datasets/598_fri_c0_1000_25.tsv",
    "datasets/599_fri_c2_1000_5.tsv",
    "datasets/601_fri_c1_250_5.tsv",
    "datasets/602_fri_c3_250_10.tsv",
    "datasets/603_fri_c0_250_50.tsv",
    "datasets/604_fri_c4_500_10.tsv",
    "datasets/605_fri_c2_250_25.tsv",
    "datasets/606_fri_c2_1000_10.tsv",
    "datasets/607_fri_c4_1000_50.tsv",
    "datasets/608_fri_c3_1000_10.tsv",
    "datasets/609_fri_c0_1000_5.tsv",
    "datasets/611_fri_c3_100_5.tsv",
    "datasets/612_fri_c1_1000_5.tsv",
    "datasets/613_fri_c3_250_5.tsv",
    "datasets/615_fri_c4_250_10.tsv",
    "datasets/616_fri_c4_500_50.tsv",
    "datasets/617_fri_c3_500_5.tsv",
    "datasets/618_fri_c3_1000_50.tsv",
    "datasets/620_fri_c1_1000_25.tsv",
    "datasets/621_fri_c0_100_10.tsv",
    "datasets/622_fri_c2_1000_50.tsv",
    "datasets/623_fri_c4_1000_10.tsv",
    "datasets/624_fri_c0_100_5.tsv",
    "datasets/626_fri_c2_500_50.tsv",
    "datasets/627_fri_c2_500_10.tsv",
    "datasets/628_fri_c3_1000_5.tsv",
    "datasets/631_fri_c1_500_5.tsv",
    "datasets/633_fri_c0_500_25.tsv",
    "datasets/634_fri_c2_100_10.tsv",
    "datasets/635_fri_c0_250_10.tsv",
    "datasets/637_fri_c1_500_50.tsv",
    "datasets/641_fri_c1_500_10.tsv",
    "datasets/643_fri_c2_500_25.tsv",
    "datasets/644_fri_c4_250_25.tsv",
    "datasets/645_fri_c3_500_50.tsv",
    "datasets/646_fri_c3_500_10.tsv",
    "datasets/647_fri_c1_250_10.tsv",
    "datasets/648_fri_c1_250_50.tsv",
    "datasets/649_fri_c0_500_5.tsv",
    "datasets/650_fri_c0_500_50.tsv",
    "datasets/651_fri_c0_100_25.tsv",
    "datasets/653_fri_c0_250_25.tsv",
    "datasets/654_fri_c0_500_10.tsv",
    "datasets/656_fri_c1_100_5.tsv",
    "datasets/657_fri_c2_250_10.tsv",
    "datasets/658_fri_c3_250_25.tsv",
    "datasets/659_sleuth_ex1714.tsv",
    "datasets/663_rabe_266.tsv",
    "datasets/665_sleuth_case2002.tsv",
    "datasets/666_rmftsa_ladata.tsv",
    "datasets/678_visualizing_environmental.tsv",
    "datasets/687_sleuth_ex1605.tsv",
    "datasets/690_visualizing_galaxy.tsv",
    "datasets/695_chatfield_4.tsv",
    "datasets/706_sleuth_case1202.tsv",
    "datasets/712_chscase_geyser1.tsv"};
    std::vector<int> dataset_size = {488,1000,1000,1000,47,50,52,108,55,200,209,1059,48,240,380,500,100,67,60,500,475,475,252,209,250,500,500,1000,500,1000,1000,1000,1000,100,1000,1000,100,1000,250,500,1000,1000,250,250,250,500,250,1000,1000,1000,1000,100,1000,250,250,500,500,1000,1000,100,1000,1000,100,500,500,1000,500,500,100,250,500,500,500,250,500,500,250,250,500,500,100,250,500,100,250,250,47,120,147,508,111,62,323,235,93,222};
    std::vector<int> dataset_vars = {4,10,4,4,13,4,2,5,2,10,6,117,4,124,2,7,2,14,15,7,3,3,14,7,5,25,25,50,25,25,100,25,50,10,25,10,5,10,5,5,25,5,5,10,50,10,25,10,50,10,5,5,5,5,10,50,5,50,25,10,50,10,5,50,10,5,5,25,10,10,50,10,25,25,50,10,10,50,5,50,25,25,10,5,10,25,7,2,6,10,3,5,4,12,6,2};
    int world_size=1,world_rank=0,name_len,TotalNRuns = 25; //MPI disabled
    //MPI_Init(NULL, NULL);
    //MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    //MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    //char processor_name[MPI_MAX_PROCESSOR_NAME];
    //MPI_Get_processor_name(processor_name, &name_len);
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
    int MaxDepth = 7;
    int NInds = 100;
    int NGens = 0;
    int MaxFEvals = 100000;
    int CrossType = 0;
    int MutType = 0;
    int Tsize = 3;
    //int max_length = 150;
    double InitProb = 0.1;
    double DSFactor = 5.0;
    double IncFactor = 0.1;
    //for(int run=0;run!=25;run++)
    for(int SelIter=6;SelIter<7;SelIter+=1)    {
    for(int P1Iter=1;P1Iter<2;P1Iter++)    {
    for(int ReplIter=3;ReplIter<4;ReplIter+=1)    {
    for(int TsizeReplIter=3;TsizeReplIter<4;TsizeReplIter++)    {
    for(int max_lengthIter=75;max_lengthIter<76;max_lengthIter+=50)    {

    DiffPRSCounter++;
    for (int RunN = 0;RunN!=TotalNRuns;RunN++)    {
        tempPRS.TaskN = TaskCounter;
        tempPRS.SelType = SelIter;
        tempPRS.ReplType = ReplIter;
        tempPRS.P1Type = P1Iter;
        tempPRS.TsizeRepl = TsizeReplIter;
        tempPRS.max_length = max_lengthIter;

        PRS.push_back(tempPRS);
        AllResults.push_back(tempResult);
        TaskFinished.push_back(0);

        string folder;
        folder = "GP18_25_Results";
        struct stat st = {0};
        if(world_rank == 0)
        {
            if (stat(folder.c_str(), &st) == -1)
                mkdir(folder.c_str(), 0777);
        }
        sprintf(buffer, "/Results_25FRONT96_S%d_C%d_M%d_R%d_P%d_T%d_TR%d_MD%d_ML%d.tsv",
            tempPRS.SelType,CrossType,MutType,tempPRS.ReplType,tempPRS.P1Type,Tsize,tempPRS.TsizeRepl,MaxDepth,tempPRS.max_length);
        tempFilePath = folder + buffer;
        FilePath.push_back(tempFilePath);
        TaskCounter++;
    }}}}}}

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

    if(world_rank == 0 && world_size > 1)
    { //MPI disabled
        /*cout<<"Rank "<<world_rank<<" starting!"<<endl;
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
                MPI_Send(&PRS[TaskToStart],PRSsize,MPI_BYTE,NodeToUse+1,0,MPI_COMM_WORLD);
                cout<<"sent task "<<TaskToStart<<" to "<<NodeToUse+1<<endl;
                NodeBusy[NodeToUse] = 1;
            }
            else
            {
                cout<<world_rank<<" Receiving result"<<endl;
                Result ReceivedRes;
                MPI_Recv(&ReceivedRes,ResultSize,MPI_BYTE,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
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
            MPI_Send(&PRSFinish,PRSsize,MPI_BYTE,i,0,MPI_COMM_WORLD);
        }*/
    }

    else
    {
        int CurTask = 0;
        while(true)
        {
            Params CurPRS;
            if(world_size > 1)
            { //MPI disabled
                /*MPI_Recv(&CurPRS,PRSsize,MPI_BYTE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                cout<<world_rank<<" received"<<endl;
                if(CurPRS.Type == -1)
                {
                    cout<<world_rank<<" Finishing!"<<endl;
                    break;
                }*/
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
                                 CurPRS.max_length,
                                 CurPRS.TsizeRepl);
                NewForest.MainLoop(Samp);
                for(int j=0;j!=ResTsize2;j++)
                {
                    ResToSend.ResultTable1[func_num][j] = ResultsArray[j];
                }
            }
            if(world_size > 1)
            { //MPI disabled
                /*cout<<world_rank<<" sending to 0 result "<<endl;
                MPI_Send(&ResToSend,ResultSize,MPI_BYTE,0,0,MPI_COMM_WORLD);
                sleep(0.1);*/
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
    //if(world_rank == 0) //MPI disabled
        //cout<<"Rank "<<world_rank<<"\ton\t"<<processor_name<<"\t Finished at"<<"\t"<<currentDateTime()<<"\n";
    //MPI_Finalize();

    t1g=clock()-t0g;
    double T0g = t1g/double(CLOCKS_PER_SEC);
    if(world_rank == 0)
    {
        cout << "Time spent: " << T0g << endl;
    }
    return 0;
}
