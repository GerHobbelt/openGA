// This library is free and distributed under
// Mozilla Public License Version 2.0.

#include <string>
#include <vector>
#include "openGA.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>

/*************************************************
This is an example of using openGA with the given
user initial solutions, based on pure arrays. 
*************************************************/

static const int ARR_SIZE = 5;

struct MySolution
{
	double* x;

	MySolution() 
	{   
		// default constructor is a must
		x = (double*) malloc(sizeof(double)*ARR_SIZE);
	}

	MySolution(double* y)
	{
		// initialize the solution with the given vector
		x = (double*) malloc(sizeof(double)*ARR_SIZE);
		for(unsigned long i=0;i<ARR_SIZE;i++)
			x[i]=y[i];
	}

	~MySolution()
	{
		if(x!=nullptr) free(x);
	}

	std::string to_string() const
	{
		std::ostringstream out;
		out<<"{";
		for(unsigned long i=0;i<ARR_SIZE;i++)
			out<<(i?",":"")<<std::setprecision(10)<<x[i];
		out<<"}";
		return out.str();
	}
};

struct MyMiddleCost
{
	// This is where the results of simulation
	// is stored but not yet finalized.
	double cost;
};

typedef EA::Genetic<MySolution,MyMiddleCost> GA_Type;
typedef EA::GenerationType<MySolution,MyMiddleCost> Generation_Type;

void init_genes(MySolution& p,const std::function<double(void)> &rnd01)
{
	for(int i=0;i<ARR_SIZE;i++)
		p.x[i] = 5.12*2.0*(rnd01()-0.5);
}

bool eval_solution(
	const MySolution& p,
	MyMiddleCost &c)
{
	constexpr double pi=3.141592653589793238;
	c.cost=10*double(ARR_SIZE);
	for(unsigned long i=0;i<ARR_SIZE;i++)
		c.cost+=p.x[i]*p.x[i]-10.0*cos(2.0*pi*p.x[i]);
	return true;
}

MySolution mutate(
	const MySolution& X_base,
	const std::function<double(void)> &rnd01,
	double shrink_scale)
{
	MySolution X_new;
	bool out_of_range;
	do{
		out_of_range=false;
		X_new=X_base;
		
		for(unsigned long i=0;i<ARR_SIZE;i++)
		{
			double mu=1.7*rnd01()*shrink_scale; // mutation radius
			X_new.x[i]+=mu*(rnd01()-rnd01());
			if(std::abs(X_new.x[i])>5.12)
				out_of_range=true;
		}
	} while(out_of_range);
	return X_new;
}

MySolution crossover(
	const MySolution& X1,
	const MySolution& X2,
	const std::function<double(void)> &rnd01)
{
	MySolution X_new;
	for(unsigned long i=0;i<ARR_SIZE;i++)
	{
		double r=rnd01();
		X_new.x[i] = r*X1.x[i]+(1.0-r)*X2.x[i];
	}
	return X_new;
}

double calculate_SO_total_fitness(const GA_Type::thisChromosomeType &X)
{
	// finalize the cost
	return X.middle_costs.cost;
}

std::ofstream output_file;

void SO_report_generation(
	int generation_number,
	const EA::GenerationType<MySolution,MyMiddleCost> &last_generation,
	const MySolution& best_genes)
{
	std::cout
		<<"Generation ["<<generation_number<<"], "
		<<"Best="<<last_generation.best_total_cost<<", "
		<<"Average="<<last_generation.average_cost<<", "
		<<"Best genes=("<<best_genes.to_string()<<")"<<", "
		<<"Exe_time="<<last_generation.exe_time
		<<std::endl;

	output_file
		<<generation_number<<"\t"
		<<last_generation.average_cost<<"\t"
		<<last_generation.best_total_cost<<"\t";
	
	for(unsigned long i=0;i<ARR_SIZE;i++)
		output_file<<best_genes.x[i]<<"\t";
		
	output_file<<"\n";
}

int main()
{
	output_file.open("./bin/result_so-rastrigin.txt");
	output_file
		<<"step"<<"\t"
		<<"cost_avg"<<"\t"
		<<"cost_best"<<"\t";
	for(unsigned long i=0;i<ARR_SIZE;i++)
		output_file<<"x_best"<<i<<"\t";
	output_file<<"\n";

	// Define some initial values (Change arrays if using other value for ARR_SIZE)
	double sol0[ARR_SIZE] = {0.0,0.0,0.0,0.0,0.0}; MySolution vsol0(sol0);
	double sol1[ARR_SIZE] = {1.0,1.0,1.0,1.0,1.0}; MySolution vsol1(sol1);
	double sol2[ARR_SIZE] = {2.0,2.0,2.0,2.0,2.0}; MySolution vsol2(sol2);	
	
	EA::Chronometer timer;
	timer.tic();

	GA_Type ga_obj;
	ga_obj.problem_mode=EA::GA_MODE::SOGA;
	ga_obj.multi_threading=true;
	ga_obj.dynamic_threading=false;
	ga_obj.idle_delay_us=0; // switch between threads quickly
	ga_obj.verbose=false;
	ga_obj.population=10;
	ga_obj.user_initial_solutions={vsol0, vsol1, vsol2};
	ga_obj.generation_max=1000;
	ga_obj.calculate_SO_total_fitness=calculate_SO_total_fitness;
	ga_obj.init_genes=init_genes;
	ga_obj.eval_solution=eval_solution;
	ga_obj.mutate=mutate;
	ga_obj.crossover=crossover;
	ga_obj.SO_report_generation=SO_report_generation;
	ga_obj.best_stall_max=20;
	ga_obj.average_stall_max=20;
	ga_obj.tol_stall_best=1e-6;
	ga_obj.tol_stall_average=1e-6;
	ga_obj.elite_count=10;
	ga_obj.crossover_fraction=0.7;
	ga_obj.mutation_rate=0.1;
	ga_obj.solve();

	std::cout<<"The problem is optimized in "<<timer.toc()<<" seconds."<<std::endl;

	output_file.close();
	return 0;
}
