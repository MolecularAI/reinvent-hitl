functions {
	real log_double_sigmoid(real value, real low, real high, real coef_div, real coef_si, real coef_se){
	real A;
	real B;
	real C;
	A = 10^(coef_se * (value / coef_div));
	B = (10^(coef_se * (value / coef_div)) + 10^(coef_se * (low / coef_div)));
	C = (10^(coef_si * (value / coef_div)) / (10^(coef_si * (value / coef_div)) + 10^(coef_si * (high / coef_div))));
	if((A / B) - C < 0)
		return 0;
	else
		return log((A / B) - C);
	}
	
	real aggregated_score_prod(vector weights,  vector x_raw, vector lows, vector deltas, vector coef_div, vector coef_si, vector coef_se, int k) {
		real score;
		score = 0;
		for (l in 1:k) {
			score = score + weights[l]*log_double_sigmoid(x_raw[l],lows[l], lows[l]+deltas[l], coef_div[l], coef_si[l], coef_se[l]);
		}
		score = exp(score);
		return score;
	}
}

data {
    int<lower=0> n;                     // number of data points
	int<lower=0> k;						// number of scoring components
	vector[k] x_raw[n];    				// raw scores
    int<lower=0,upper=1> y[n];          // binary response variable
	vector<lower=0>[k] weights;			// Non-negative weights of the user-model (fixed, sum 1)
	vector[k] coef_div;					// params of double sigmoid (fixed)
	vector[k] coef_si;					// params of double sigmoid (fixed)
	vector[k] coef_se;					// params of double sigmoid (fixed)
	vector[k] high0;					// initial guesses of the score transform variables
	vector[k] low0;
	int<lower=0> npred;
	vector[k] xpred[npred];    // all molecules (for active learning)
}

parameters {
	vector[k] lows;
	vector<lower=0>[k] deltas;
}
model {
	for (i in 1:k) {
		lows[i] ~ normal(low0[i], (high0[i]-low0[i])/8);
		deltas[i] ~ normal(high0[i]-low0[i], (high0[i]-low0[i])/8);
	}
    // observation model
	for (j in 1:n) {
		y[j] ~ bernoulli(aggregated_score_prod(weights, x_raw[j], lows, deltas, coef_div, coef_si, coef_se, k));
	}
}
generated quantities {
	vector[k] highs;
	vector[npred] score_pred;
	for (i in 1:k)
		highs[i] = lows[i]+deltas[i];
	for (j in 1:npred){
		score_pred[j] = aggregated_score_prod(weights, xpred[j], lows, deltas, coef_div, coef_si, coef_se, k);
	}
}