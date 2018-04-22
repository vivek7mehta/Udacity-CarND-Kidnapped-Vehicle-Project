/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 10;


    normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

    for(int i=0;i<num_particles;i++)
    {
        Particle p;
        p.id=i;
        p.weight = 1;

        double sample_x, sample_y, sample_theta;
        default_random_engine gen;
        sample_x = dist_x(gen);
        sample_y = dist_y(gen);
        sample_theta = dist_theta(gen);

        p.x = sample_x;
        p.y = sample_y;
        p.theta = sample_theta;
        particles.push_back(p);
        weights.push_back(p.weight);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;

    for(int i=0;i<num_particles;i++)
    {
        normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
        normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
        normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);

        if(fabs(yaw_rate)<0.0001)
        {
            particles[i].x = dist_x(gen);
            particles[i].y = dist_y(gen);

            particles[i].x += velocity*delta_t*cos(particles[i].theta);
            particles[i].y += velocity*delta_t*sin(particles[i].theta);

        }else{
            particles[i].x = dist_x(gen);
            particles[i].y = dist_y(gen);
            particles[i].theta = dist_theta(gen);

            particles[i].x += velocity*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta))/yaw_rate;
            particles[i].y += velocity*(cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t))/yaw_rate;
            particles[i].theta += yaw_rate*delta_t;
        }
    }

}

void ParticleFilter::dataAssociation(int particle_index,std::vector<LandmarkObs> predicted, std::vector<Map::single_landmark_s> landmark_list) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

        particles[particle_index].associations.clear();
        particles[particle_index].sense_x.clear();
        particles[particle_index].sense_y.clear();
        for(int j=0;j<predicted.size();j++)
        {
            double min_dist = numeric_limits<double>::max();
            int min_dist_landmark_index = 0;
            for(int k=0;k<landmark_list.size();k++)
            {
                 double d = dist(landmark_list[k].x_f,landmark_list[k].y_f,predicted[j].x,predicted[j].y);
                 if(d<min_dist){min_dist_landmark_index = k;min_dist=d;}
            }
            particles[particle_index].associations.push_back(landmark_list[min_dist_landmark_index].id_i);
            particles[particle_index].sense_x.push_back(predicted[j].x);
            particles[particle_index].sense_y.push_back(predicted[j].y);
        }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    for(int i=0;i<num_particles;i++){

        // Transformation
        vector<LandmarkObs> tObs;
        for(int j=0;j<observations.size();j++)
        {
            LandmarkObs l;
            l.id = observations[j].id;
            l.x = particles[i].x + (cos(particles[i].theta) * observations[j].x) - (sin(particles[i].theta) * observations[j].y);
            l.y = particles[i].y + (sin(particles[i].theta) * observations[j].x) + (cos(particles[i].theta) * observations[j].y);
            tObs.push_back(l);
        }
        dataAssociation(i,tObs,map_landmarks.landmark_list);

        weights[i]=1;
        for(int j=0;j<particles[i].associations.size();j++){
            double gauss_norm= (1/(2 * M_PI * std_landmark[0] * std_landmark[1]));

            int land_ind = 0;
            for(int k=0;k<map_landmarks.landmark_list.size();k++)
            {
                if(particles[i].associations[j]==map_landmarks.landmark_list[k].id_i){land_ind=k;break;};
            }
            double mu_x = map_landmarks.landmark_list[land_ind].x_f;
            double mu_y = map_landmarks.landmark_list[land_ind].y_f;
            double x_obs = tObs[j].x;
            double y_obs = tObs[j].y;
            double exponent= ((x_obs - mu_x)*(x_obs - mu_x))/(2 * std_landmark[0]*std_landmark[0]) + ((y_obs - mu_y)*(y_obs - mu_y))/(2 * std_landmark[1]*std_landmark[1]);
            weights[i]*= gauss_norm * exp(-exponent);
        }
    }

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution


    default_random_engine gen;
    std::discrete_distribution<> d(weights.begin(),weights.end());
    vector<Particle> p2;
    double new_weight_sum = 0;
    for(int i=0;i<num_particles;i++)
    {
        p2.push_back(particles[d(gen)]);
        weights[i]  = p2[i].weight;
        new_weight_sum += weights[i];
    }
    //normalize
    for(int i=0;i<num_particles;i++)
    {
        weights[i] /= new_weight_sum;
        p2[i].weight = weights[i];
    }
    particles = p2;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
