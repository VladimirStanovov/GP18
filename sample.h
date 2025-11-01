#include <math.h>
#include <iostream>
#include <time.h>
#include <fstream>
#include <random>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
//#include <conio.h>
#include <string.h>

using namespace std;

typedef std::chrono::high_resolution_clock myclock;
myclock::time_point beginning = myclock::now();
myclock::duration d1 = myclock::now() - beginning;
//unsigned globalseed = d1.count();
unsigned globalseed = unsigned(time(NULL));
//unsigned globalseed = 2018;
unsigned seed1 = globalseed+0;
unsigned seed2 = globalseed+100;
unsigned seed3 = globalseed+200;
unsigned seed4 = globalseed+300;
unsigned seed5 = globalseed+400;
std::mt19937 generator_uni_i(seed1);
std::mt19937 generator_uni_r(seed2);
std::mt19937 generator_norm(seed3);
std::mt19937 generator_cachy(seed4);
std::mt19937 generator_uni_i_2(seed5);
std::uniform_int_distribution<int> uni_int(0,32768);
std::uniform_real_distribution<double> uni_real(0.0,1.0);
std::normal_distribution<double> norm_dist(0.0,1.0);
std::cauchy_distribution<double> cachy_dist(0.0,1.0);

unsigned long mix(unsigned long a, unsigned long b, unsigned long c)
{
    a=a-b;  a=a-c;  a=a^(c >> 13);
    b=b-c;  b=b-a;  b=b^(a << 8);
    c=c-a;  c=c-b;  c=c^(b >> 13);
    a=a-b;  a=a-c;  a=a^(c >> 12);
    b=b-c;  b=b-a;  b=b^(a << 16);
    c=c-a;  c=c-b;  c=c^(b >> 5);
    a=a-b;  a=a-c;  a=a^(c >> 3);
    b=b-c;  b=b-a;  b=b^(a << 10);
    c=c-a;  c=c-b;  c=c^(b >> 15);
    return c;
}

int IntRandom(const int target)
{
    if(target == 0)
        return 0;
    return uni_int(generator_uni_i)%target;
}
double Random(const double minimal, const double maximal)
{
    return uni_real(generator_uni_r)*(maximal-minimal)+minimal;
}
double NormRand(const double mu, const double sigma)
{
    return norm_dist(generator_norm)*sigma + mu;
}
double CauchyRand(const double mu, const double sigma)
{
    return cachy_dist(generator_cachy)*sigma+mu;
}
void qSort1(double* Mass, const int low, const int high)
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
            i++;    j--;
        }
    } while(i<=j);
    if(low<j)   qSort1(Mass,low,j);
    if(i<high)  qSort1(Mass,i,high);
}

class sample
{
public:
  //конструктор для задач классификации
  sample();
  ~sample();
  void Init(int NewSize, int NewNVars, int NewNClasses, int NewNFolds,
         double NewSplitRate, int NewProblemType);
  void CleanSamp();
  //задание значений в выборке
  void SetValue(int Num, int Var, double value);
  void SetNormValue(int Num, int Var, double value);
  void SetOut(int Num, int Out, double value);
  void SetClass(int Num, int Class);
  //задание положений пропущенных значений
  void SetMissingInput(int Num, int Var);
  void SetMissingOutput(int Num, int Out);
  //первичное считывание с файла
  void ReadFileClassification(char* filename);
  void ReadFileRegression(char* filename);
  void ReadFileRegression_SRBENCH(char* filename);
  //вывод на экран всей выборки
  void ShowSampleClassification();
  void ShowNormSampleClassification();
  void ShowSampleRegression();
  //взять значение переменной из выборки
  double GetValue(int Num,int Var);
  //взять значение переменной из выборки
  double GetNormValue(int Num,int Var);
  //взять значение выхода из выборки
  double GetOutput(int Num,int Var);
  //получить номер класса для измерения
  int GetClass(int Num);
  //разбиение выборки, кросс-валидация
  void SplitCVRandom();
  void SplitCVStratified();
  //считает число объектов по классам
  void ClassPatternsCalc();
  //простое разбиение выборки
  void SplitRandom();
  void SplitStratified();
  //возвращает объем обучающей выборки для кросс-валидации
  int GetCVLearnSize(int FoldOnTest);
  //возвращает объем тестовой выборки для кросс-валидации
  int GetCVTestSize(int FoldOnTest);
  //возвращает объем обучающей выборки
  int GetLearnSize();
  //возвращает объем тестовой выборки
  int GetTestSize();
  //вернуть номер кросс-валидационного разбиения
  int GetCVFoldNum(int Num);
  //вернуть число переменных
  int GetNVars();
  //вернуть число классов
  int GetNClasses();
  //вернуть размер выборки
  int GetSize();

  int GetClassPerFold(int ClassNum,int FoldNum);

  int GetClassPositions(int ClassNum,int Num);

  int GetNClassInst(int ClassNum);
  //Задать обучающую выборку, кросс-валидация
  void SetCVLearn(sample &S_CVLearn, int FoldOnTest);
  //Задать тестовую выборку, кросс-валидация
  void SetCVTest(sample &S_CVTest, int FoldOnTest);
  //Задать обучающую выборку
  void SetLearn(sample& S_Learn);
  //Задать тестовую выборку
  void SetTest(sample& S_Test);
  //нормализация выборки на [0,1]
  void NormalizeCV_01(int FoldOnTest);

  //общие параметры
  int Size;         //объем выборки
  int NCols;        //общее число столбцов в выборке
  int NVars;        //число столбцов входных параметров
  int NOuts;        //число столбцов выходных параметров
  int ProblemType;  //тип задачи
  int NFolds;       //число частей при кросс-валидации
  double SplitRate; //доля обучающей выборки, например
                    //0.7 => разбиение 70/30
  int LearnSize;    //размер обучающей выборки
  int TestSize;     //размер тестовой выборки

  double** Inputs;  //входы задачи
  double** NormInputs; //Нормализованные входы задачи
  double** Outputs; //выходы задачи
  bool** MissingInputs;    //массив пропущенных входных значений
  bool** MissingOutputs;   //массив пропущенных выходных значений
  int* FoldSize;    //размеры частей, на которые разбивается
                    //выборка при кросс-валидации
  int* CVFoldNum;   //номер части, к которой относится измерение

  //параметры для задач классификации
  int NClasses;     //число классов в задаче
  int* Classes;     //массив номеров классов
  int* NClassInst;  //количество объектов в классах
  int** ClassPositions;  //Номера объектов, принадлежащих разным классам
  int** ClassPerFold;    //число объектов классов для каждой части
  double** Range;   //диапазоны переменных для нормализации

};
