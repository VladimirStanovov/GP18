#include "sample.h"

sample::sample()
{

}
void sample::Init(int NewSize, int NewNVars, int NewNClasses, int NewNFolds,
               double NewSplitRate, int NewProblemType)
{
  Size = NewSize;
  NClasses = NewNClasses;
  NVars = NewNVars;
  ProblemType = NewProblemType;  //0 - классификация, 1 - регрессия
  NFolds = NewNFolds;
  SplitRate = NewSplitRate;
  NOuts = 1;

  ClassPerFold = new int*[NClasses];
  ClassPositions = new int*[NClasses];
  CVFoldNum = new int[Size];
  FoldSize = new int[NFolds];
  NClassInst = new int[NClasses];
  Classes = new int[Size];
  Inputs = new double*[Size];
  Outputs = new double*[Size];
  MissingInputs = new bool*[Size];
  NormInputs = new double*[Size];
  for(int i=0;i!=Size;i++)
  {
    Outputs[i] = new double[NOuts];
    Inputs[i] = new double[NVars];
    MissingInputs[i] = new bool[NVars];
    for(int j=0;j!=NVars;j++)
    {
        MissingInputs[i][j] = false;
    }
    NormInputs[i] = new double[NVars];
  }
  for(int i=0;i!=NClasses;i++)
  {
      ClassPositions[i] = new int[Size];
      ClassPerFold[i] = new int[NFolds];
  }
  Range = new double*[NVars];
  for(int i=0;i!=NVars;i++)
  {
      Range[i] = new double[2];
  }
}
sample::~sample()
{

}
void sample::CleanSamp()
{
  delete FoldSize;
  for(int i=0;i!=Size;i++)
  {
    delete Inputs[i];
    delete MissingInputs[i];
    delete NormInputs[i];
  }
  delete Inputs;
  delete MissingInputs;
  delete NormInputs;
  if(ProblemType == 1)
  {
        for(int i=0;i!=Size;i++)
        {
            delete Outputs[i];
            delete MissingOutputs[i];
        }
        delete Outputs;
        delete MissingOutputs;
  }

  if(ProblemType == 0)
  {
    for(int i=0;i!=NVars;i++)
    {
        delete Range[i];
    }
    delete Range;
    delete Classes;
    delete NClassInst;
    for(int i=0;i!=NClasses;i++)
    {
      delete ClassPositions[i];
      delete ClassPerFold[i];
    }
    delete ClassPositions;
    delete ClassPerFold;
  }
}
void sample::ReadFileClassification(char* filename)
{
   std::ifstream fin(filename);
   char tempstring [80];
   for(int i=0;i!=Size;i++)
   {
       for(int j=0;j!=NVars;j++)
       {
          fin>>tempstring;
          if(strcmp(tempstring,"?") == 0)
          {
              MissingInputs[i][j] = true;
              Inputs[i][j] = 0;
          }
          else
          {
              Inputs[i][j] = atof(tempstring);
          }
            //fin>>Inputs[i][j];
          //cout<<Inputs[i][j]<<"\t";
       }
       fin>>tempstring;
       if(strcmp(tempstring,"?") == 0)
       {
            cout<<i<<" - WRONG!"<<endl;
            //_getch();
       }
       else
       {
           Classes[i] = atoi(tempstring);
       }
       //fin>>Classes[i];

       //cout<<Classes[i];
       //cout<<endl;
       //if(Classes[i] > 1)
       //{
       //     cout<<i<<" - WRONG!"<<endl;
       //     _getch();
       //}
   }
}
void sample::ReadFileRegression(char* filename)
{
   std::ifstream fin(filename);
   char tempstring [80];
   for(int i=0;i!=Size;i++)
   {
       for(int j=0;j!=NVars;j++)
       {
          fin>>tempstring;
          if(strcmp(tempstring,"?") == 0)
          {
              MissingInputs[i][j] = true;
              Inputs[i][j] = 0;
          }
          else
          {
              Inputs[i][j] = atof(tempstring);
          }
       }
       for(int j=0;j!=NOuts;j++)
       {
          fin>>tempstring;
          if(strcmp(tempstring,"?") == 0)
          {
              MissingOutputs[i][j] = true;
              Outputs[i][j] = 0;
          }
          else
          {
              Outputs[i][j] = atof(tempstring);
          }
       }
   }
}
void sample::ReadFileRegression_SRBENCH(char* filename)
{
   std::ifstream fin(filename);
   string line;
   std::getline(fin, line);
   char tempstring [80];
   for(int i=0;i!=Size;i++)
   {
       for(int j=0;j!=NVars;j++)
       {
          fin>>tempstring;
          if(strcmp(tempstring,"?") == 0)
          {
              MissingInputs[i][j] = true;
              Inputs[i][j] = 0;
          }
          else
          {
              Inputs[i][j] = atof(tempstring);
          }
       }
       for(int j=0;j!=NOuts;j++)
       {
          fin>>tempstring;
          if(strcmp(tempstring,"?") == 0)
          {
              MissingOutputs[i][j] = true;
              Outputs[i][j] = 0;
          }
          else
          {
              Outputs[i][j] = atof(tempstring);
          }
       }
   }
}
void sample::SetValue(int Num, int Var, double value)
{
  Inputs[Num][Var] = value;
}
void sample::SetNormValue(int Num, int Var, double value)
{
  NormInputs[Num][Var] = value;
}
void sample::SetOut(int Num, int Out, double value)
{
  Outputs[Num][Out] = value;
}
void sample::SetClass(int Num, int Class)
{
  Classes[Num] = Class;
}
void sample::SetMissingInput(int Num, int Var)
{
  MissingInputs[Num][Var] = true;
}
void sample::SetMissingOutput(int Num, int Out)
{
  MissingOutputs[Num][Out] = true;
}
void sample::ShowSampleRegression()
{
  for(int i=0;i!=Size;i++)
  {
      cout<<i<<":\t";
      for(int j=0;j!=NVars;j++)
      {
          if(MissingInputs[i][j])
            cout<<"?\t";
          else
            cout<<Inputs[i][j]<<"\t";
      }
      cout<<"->\t";
      for(int j=0;j!=NOuts;j++)
      {
          if(MissingOutputs[i][j])
            cout<<"?\t";
          else
            cout<<Outputs[i][j]<<"\t";
      }
      cout<<endl;
  }
}
void sample::ShowSampleClassification()
{
  for(int i=0;i!=Size;i++)
  {
      cout<<i<<":\t";
      for(int j=0;j!=NVars;j++)
      {
          if(MissingInputs[i][j])
            cout<<"?\t";
          else
            cout<<Inputs[i][j]<<"\t";
      }
      cout<<"->\t";
      cout<<Classes[i]<<endl;
  }
}
void sample::ShowNormSampleClassification()
{
  for(int i=0;i!=Size;i++)
  {
      cout<<i<<":\t";
      for(int j=0;j!=NVars;j++)
      {
          if(MissingInputs[i][j])
            cout<<"?\t";
          else
            cout<<NormInputs[i][j]<<"\t";
      }
      cout<<"->\t";
      cout<<Classes[i]<<endl;
  }
}
double sample::GetValue(int Num,int Var)
{
  return Inputs[Num][Var];
}
double sample::GetNormValue(int Num,int Var)
{
  return NormInputs[Num][Var];
}
double sample::GetOutput(int Num,int Var)
{
  return Outputs[Num][Var];
}
int sample::GetClass(int Num)
{
  return Classes[Num];
}
void sample::SplitCVStratified()
{
  int RandomPattern;
  for(int i=0;i!=Size;i++)
  {
    CVFoldNum[i] = -1;
  }
  for(int i=0;i!=NFolds;i++)
  {
      FoldSize[i] = 0;
      for(int j=0;j!=NClasses;j++)
      {
          //cout<<ClassPerFold[j][i]<<"\t";
         for(int k=0;k!=ClassPerFold[j][i];k++)
         {
            RandomPattern = IntRandom(NClassInst[j]);
            while(CVFoldNum[ClassPositions[j][RandomPattern]] != -1)
            {
              RandomPattern++;
              if(RandomPattern == NClassInst[j])
                RandomPattern=0;
            }
            CVFoldNum[ClassPositions[j][RandomPattern]] = i;
            FoldSize[i]++;
          }
      }
      //cout<<endl;
  }
}
void sample::SplitCVRandom()
{
  int counter=0;
  int RandomPattern;
  for(int i=0;i!=Size;i++)
  {
    CVFoldNum[i] = -1;
  }
  for(int i=0;i!=NFolds;i++)
  {
    FoldSize[i] = 0;
    while(counter < (int) ((double)i+1.0)*(double)Size/(double)NFolds )
    {
      RandomPattern = IntRandom(Size);
      while(CVFoldNum[RandomPattern] != -1) //Если измерение уже взято в
      {                                     //одну из выборок, то увеличиваем
        RandomPattern++;                    //номер, пока не найдем не взятое
        if(RandomPattern == Size)           //таким образом избегаем смещения
          RandomPattern=0;                  //если достигли конца, начинаем
      }                                     //сначала.

      CVFoldNum[RandomPattern] = i;
      counter++;
      FoldSize[i]++;
    }
  }
}
void sample::ClassPatternsCalc()
{
  for(int i=0;i!=NClasses;i++)
  {
      NClassInst[i] = 0;
  }
  for(int i=0;i!=Size;i++)
  {
      NClassInst[GetClass(i)]++;
  }
  for(int i=0;i!=NClasses;i++)
  {
      int counter = 0;
      for(int j=0;j!=Size;j++)
      {
          if(GetClass(j) == i)
          {
              ClassPositions[i][counter] = j;
              counter++;
          }
      }
      counter = 0;
      for(int j=0;j!=NFolds;j++)
      {
          ClassPerFold[i][j] = 0;
          while(counter < int( ((double)j+1.0)*(double)NClassInst[i]/(double)NFolds ) )
          {
              ClassPerFold[i][j] ++;
              counter++;
          }
      }
  }
}
void sample::SplitRandom()
{
  int RandomPattern;
  for(int i=0;i!=Size;i++)
  {
    CVFoldNum[i] = -1;  //используется здесь как индикатор
                        //использования измерения
  }
  LearnSize = int(SplitRate*(double)Size);
  TestSize = Size-LearnSize;
  for(int i=0;i!=LearnSize;i++)
  {
    RandomPattern = IntRandom(Size);
    while(CVFoldNum[RandomPattern] != -1)
    {
      RandomPattern++;
      if(RandomPattern == Size)
        RandomPattern=0;
    }
    CVFoldNum[RandomPattern] = 0;
  }
  for(int i=0;i!=TestSize;i++)
  {
    RandomPattern = IntRandom(Size);
    while(CVFoldNum[RandomPattern] != -1)
    {
      RandomPattern++;
      if(RandomPattern == Size)
        RandomPattern=0;
    }
    CVFoldNum[RandomPattern] = 1;
  }
}
void sample::SplitStratified()
{
  int counter=0;
  int RandomPattern;
  for(int i=0;i!=Size;i++)
  {
    CVFoldNum[i] = -1;  //используется здесь как индикатор
                        //использования измерения
  }
  for(int i=0;i!=NClasses;i++)
  {
    for(int j=0;j!=round(SplitRate*NClassInst[i]);j++)
    {
      RandomPattern = IntRandom(NClassInst[i]);
      while(CVFoldNum[ClassPositions[i][RandomPattern]] != -1)
      {
        RandomPattern++;
        if(RandomPattern == NClassInst[i])
          RandomPattern=0;
      }
      CVFoldNum[ClassPositions[i][RandomPattern]] = 0;
      counter++;
    }
  }
  LearnSize = counter;
  TestSize = Size-LearnSize;
  for(int i=0;i!=Size;i++)
  {
    if(CVFoldNum[i] == -1)
      CVFoldNum[i] = 1;
  }
}
int sample::GetTestSize()
{
  return TestSize;
}
int sample::GetLearnSize()
{
  return LearnSize;
}
int sample::GetCVLearnSize(int FoldOnTest)
{
  LearnSize=0;
  for(int i=0;i!=NFolds;i++)
  {
    if(i != FoldOnTest)
      LearnSize += FoldSize[i];
  }
  return LearnSize;
}
int sample::GetCVTestSize(int FoldOnTest)
{
  TestSize=FoldSize[FoldOnTest];
  return TestSize;
}
int sample::GetCVFoldNum(int Num)
{
    return CVFoldNum[Num];
}
int sample::GetNVars()
{
    return NVars;
}
int sample::GetNClasses()
{
    return NClasses;
}
int sample::GetNClassInst(int ClassNum)
{
    return NClassInst[ClassNum];
}
int sample::GetSize()
{
    return Size;
}
int sample::GetClassPerFold(int ClassNum,int FoldNum)
{
    return ClassPerFold[ClassNum][FoldNum];
}
int sample::GetClassPositions(int ClassNum,int Num)
{
    return ClassPositions[ClassNum][Num];
}
void sample::SetCVLearn(sample &S_CVLearn, int FoldOnTest)
{
  int counter=0;
  for(int i=0;i!=Size;i++)
  {
    if(CVFoldNum[i] != FoldOnTest)
    {
      for(int j=0;j!=NVars;j++)
      {
        S_CVLearn.SetValue(counter,j,GetValue(i,j));
        if(MissingInputs[i][j])
          S_CVLearn.SetMissingInput(counter,j);
      }
      if(ProblemType == 0)
      {
        S_CVLearn.SetClass(counter,GetClass(i));
      }
      else
      {
        for(int j=0;j!=NOuts;j++)
        {
          S_CVLearn.SetOut(counter,j,GetOutput(i,j));
          if(MissingOutputs[i][j])
            S_CVLearn.SetMissingOutput(counter,j);
        }
      }
      counter++;
    }
  }
}
void sample::SetCVTest(sample &S_CVTest, int FoldOnTest)
{
  int counter=0;
  for(int i=0;i!=Size;i++)
  {
    if(CVFoldNum[i] == FoldOnTest)
    {
      for(int j=0;j!=NVars;j++)
      {
        S_CVTest.SetValue(counter,j,GetValue(i,j));
        if(MissingInputs[i][j])
          S_CVTest.SetMissingInput(counter,j);
      }
      if(ProblemType == 0)
      {
        S_CVTest.SetClass(counter,GetClass(i));
      }
      else
      {
        for(int j=0;j!=NOuts;j++)
        {
          S_CVTest.SetOut(counter,j,GetOutput(i,j));
          if(MissingOutputs[i][j])
            S_CVTest.SetMissingOutput(counter,j);
        }
      }
      counter++;
    }
  }
}
void sample::SetLearn(sample& S_Learn)
{
  int counter=0;
  for(int i=0;i!=Size;i++)
  {
    if(CVFoldNum[i] == 0)
    {
      for(int j=0;j!=NVars;j++)
      {
        S_Learn.SetValue(counter,j,GetValue(i,j));
        if(MissingInputs[i][j])
          S_Learn.SetMissingInput(counter,j);
      }
      if(ProblemType == 0)
      {
        S_Learn.SetClass(counter,GetClass(i));
      }
      else
      {
        for(int j=0;j!=NOuts;j++)
        {
          S_Learn.SetOut(counter,j,GetOutput(i,j));
          if(MissingOutputs[i][j])
            S_Learn.SetMissingOutput(counter,j);
        }
      }
      counter++;
    }
  }
}
void sample::SetTest(sample& S_Test)
{
  int counter=0;
  for(int i=0;i!=Size;i++)
  {
    if(CVFoldNum[i] == 1)
    {
      for(int j=0;j!=NVars;j++)
      {
        S_Test.SetValue(counter,j,GetValue(i,j));
        if(MissingInputs[i][j])
          S_Test.SetMissingInput(counter,j);
      }
      if(ProblemType == 0)
      {
        S_Test.SetClass(counter,GetClass(i));
      }
      else
      {
        for(int j=0;j!=NOuts;j++)
        {
          S_Test.SetOut(counter,j,GetOutput(i,j));
          if(MissingOutputs[i][j])
            S_Test.SetMissingOutput(counter,j);
        }
      }
      counter++;
    }
  }
}
void sample::NormalizeCV_01(int FoldOnTest)
{
    for(int i=0;i!=NVars;i++)
    {
        Range[i][0] = GetValue(0,i);
        Range[i][1] = GetValue(0,i);
        for(int j=1;j!=Size;j++)
        {
            if(!MissingInputs[j][i])
            {
                if(GetCVFoldNum(j) != FoldOnTest)
                {
                    //cout<<j<<"\t"<<i<<"\t"<<Range[i][0]<<"\t"<<GetValue(j,i)<<"\t"<<Range[i][1]<<endl;
                    if(Range[i][0] < GetValue(j,i))
                        Range[i][0] = GetValue(j,i);
                    if(Range[i][1] > GetValue(j,i))
                        Range[i][1] = GetValue(j,i);
                    //cout<<j<<"\t"<<i<<"\t"<<Range[i][0]<<"\t"<<GetValue(j,i)<<"\t"<<Range[i][1]<<endl;
                }
            }
        }
        if(Range[i][0] == Range[i][1])
            Range[i][1] = Range[i][0] + 1;
        //cout<<Range[i][0]<<"\t"<<Range[i][1]<<endl;
    }
    for(int i=0;i!=Size;i++)
    {
        for(int j=0;j!=NVars;j++)
        {
            if(MissingInputs[i][j] == 0)
            {
                SetNormValue(i,j,(GetValue(i,j)-Range[j][1])/(Range[j][0]-Range[j][1]));
                //cout<<NormInputs[i][j]<<"\t";
            }
        }
        //cout<<endl;
    }
}
