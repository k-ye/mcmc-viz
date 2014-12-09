MCMC Visualization
-
**Author** 

Dustin Tran \<dtran@g.harvard.edu\>

Ye Kuang \<yekuang@g.harvard.edu\>

-
### Background ###
This is a self research prompt project. In this project we are designing a visualizer to reveal the sampling process and empirical posterior distribution surface for the MCMC process. The MCMC sampler supports any kind of dimension of parameters (as long as you implement the proposal and posterior function correctly, of course), but for visualization, we will only display parameters in the first two dimension.

### Features ###
- Supports user defined sampling functions(proposal) and models(posterior). 
- Switch between 3D posterior surface and 2D heatmap, which represents the contour by pressing `F`.
- Switch to display either number of iterations or acceptance rate in the right-bottom lable by pressing `T`. Or you could just switch off the display.
- Will display the latest 25 sampling point trajectory. Helps you check if your proposal has stuck in certain local region.

### Get Started ###
In terminal, type in `./MCMC` to see the posterior surface. Try the features mentioned above to get a basic idea of your model.

### Plugin Your Own Model ###
Three steps need to be done in order to use the MCMC visualizer. 

- Define the class to propose new parameters and to calculate the posterior according to your statistical model.
- Define the approximated range of your parameter space and the initial parameter
- Make a small change in `ProblemFactory` namespace so that your model can be correctly linked with our simulator and visualizer.
 
First of all, you will provide a proposal function and a posterior function that are specific related to your statistical model first with our pre-defined interface like this.
```
typedef double value_type;
typedef std::vector<value_type> param_type;

class YourModelProposal {
	/* Function to propose a random parameter that is of dimension @a prev_param.size(). This is where you propose new random parameters.
	 * @param[in] prev_param: the previous parameter in the MCMC chain
	 * @param[in] iteration: current iteration
	 * @result: new proposed paramter
	 */
	param_type rand(const param_type& prev_param, unsigned iteration) {
		// propose a new random parameter...
	}

	/* Function to calculate the probability density of the state transition from @a param_from to @a param_to. This is useful in determining the acceptance ratio for the new proposed parameter.
	 * @param[in] param_from, param_to: previous and new state of the parameter
	 * @param[in] log_: specify if returning density or log density
	 * @result: density of the transition
	 */
	value_type dens(const param_type& param_to, const param_type& param_from, bool log_) {
	}
};

class YourModelPosterior {
	/* Function to calculate the posterior density given the current parameter under the statistical model.
	 * @param[in] param: current parameter
	 * @param[in] log_: specify if returning density or log density
	 * @result: density of @a param under the model
	 */
	value_type operator()(const param_type& param, bool log_) {
	}
};
```
***Hint***: If you don't know how write out the <i>p.d.f</i> you are using elegantly, take a look at the `boost::math` and `boost::random` library. Most of the distributions have already been built up here for you! Specifications regarding the libarries can be found [here](http://www.boost.org/doc/libs/1_57_0/libs/math/doc/html/dist.html).

Once you define your own problem statistical model (the Posterior) and proposal function (the Proposal), you can give an estimation of your parameter set and define its range in another class. This is necessary as the visualizer needs to find a way to map the parameters back onto the displayed mesh coordination. Also, you will need to provide the initial parameter for the simulation.

```
struct YourModelRange {
	// Range of your first dimension parameter, [@a x.first, @a x.second]
	std::pair<value_type, value_type> x;
	// Range of your second dimension parameter, [@a y.first, @a y.second]
	std::pair<value_type, value_type> y;
};

struct YourModelInitialParams {
	/* Return the initial parametes
	 */
	param_type initial_params() {
		return {100., .5};
	}
}
```

Finally, we would like to tell `ProblemFactory` where it can find all these defined classes. So inside `ProblemFactory.hpp`, you need to do the following changes.
```
#include "YourModel.hpp"

// define your model types into the type names used by visualizer
typedef YourModel::YourModelProposal ProblemProposal;
typedef YourModel::YourModelPosterior ProblemPosterior;
typedef YourModel::YourModelRange ProblemRange;
typedef YourModel::YourModelInitialParams ProblemInitialParams;
```

In order to make use of the problem factory, you may wrap all the classes related to your model in `YourModel` namespace.

### Example ###

In the example we provided with two sample model.

The first model, `SimpleProblem.hpp`, is the mixture of three bi-variate normal distributions with their centers on the three vertices of a equilateral triangle. We are using another bi-variate normal distribution as the proposal function, whose center is also the center of that triangle.


![](Demo/simple_1.png) 
![](Demo/simple_2.png)

The second model, `Waterbuck.hpp`, is more complicated. We are using a series of data that are generated from a binomial distribution (N, p), without knowing any prior of the parameters. Still, we use two uni-variate normal distribution as proposal functions. From the figure we can see that this is not an ideal proposal function, as the sampler is always trapped in certain area, with a slow speed to traverse through the entire parameter space.

![](Demo/waterbuck_1.png)
![](Demo/waterbuck_2.png)
![](Demo/waterbuck_3.png)
