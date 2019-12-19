'''
# ------------------------------------------------------------------------
# NOTE
# ------------------------------------------------------------------------
# Purpose:
# Visualize discrete choice to understand how parameters affect results
# Individual choice given utility 
# 
# methods include:
# ------------------------------------------------------------------------
'''

import numpy as np
import pandas as pd
import os,sys,inspect
import scipy.stats as stats
import scipy.optimize as opt
import scipy.integrate as integrate
from scipy.io import loadmat
import econtools 
import econtools.metrics as mt
import statsmodels.discrete.discrete_model as sm
import matplotlib.pyplot as plt
import itertools as it
import copy




# ---------------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------------

class one_market:
    '''A toy for discrete choice with heterogeneous consumers:
    Taste: uniform distributed, range = sigma, mean = beta 
    Price sensitivity: alpha, no randomness '''

    def __init__(self, alpha, beta , sigma):
        '''input random coefficients '''

        self.construct_product_char()
        self.construct_consumer_types(alpha, beta, sigma)

        return 

    # Main Function ----------------------------------------------------

    def baseline_plot(self):
        
        fig = plt.figure(figsize=(10,5)) 
        self.plot_type_choices()

        # title
        fig.legend(loc = 'lower right')
        fig.text(0, -0.1, 
            'Parameters: price sensitivity alpha = {}, taste on X mean = {}, variance = {}, range = [{}, {}]'.format(
                self.consumer_alpha, 
                self.consumer_beta, 
                self.consumer_sigma,
                self.consumer_beta-self.consumer_sigma*0.5,
                self.consumer_beta+self.consumer_sigma*0.5), 
            fontsize = 10)

        plt.close()

        return fig

    def price_or_quality_change(self, delta_X, delta_p):

        # 1. baseline figure
        fig = plt.figure(figsize=(10,5)) 
        self.plot_type_choices()
        fig.legend(loc = 'lower right')
        fig.text(0, -0.1, 
            'Parameters: price sensitivity alpha = {}, taste on X mean = {}, variance = {}, range = [{}, {}]'.format(
                self.consumer_alpha, 
                self.consumer_beta, 
                self.consumer_sigma,
                self.consumer_beta-self.consumer_sigma*0.5,
                self.consumer_beta+self.consumer_sigma*0.5), 
            fontsize = 10)

        # 2 new figure
        self.construct_product_char(delta_X, delta_p)
        self.plot_type_choices(baseline=False)
        fig.text(0, -0.15, 
            'Low quality good change: X increase {}, price increase {}; High quality good change: X increase {}, price increase {}'.format(delta_X[0], delta_p[0], delta_X[1], delta_p[1]), 
            fontsize = 10)

        # set the product_character back
        self.construct_product_char()
        
        plt.close()

        return fig

    def taste_change(self, alpha, beta, sigma):

        # 1. baseline figure
        fig = plt.figure(figsize=(10,5)) 
        self.plot_type_choices()
        fig.legend(loc = 'lower right')
        fig.text(0, -0.1, 
            'Old Parameters: price sensitivity alpha = {}, taste on X mean = {}, variance = {}, range = [{}, {}]'.format(
                self.consumer_alpha, 
                self.consumer_beta, 
                self.consumer_sigma,
                self.consumer_beta-self.consumer_sigma*0.5,
                self.consumer_beta+self.consumer_sigma*0.5), 
            fontsize = 10)

        # 2. new figure
        old_par = [self.consumer_alpha, self.consumer_beta, self.consumer_sigma]
        self.construct_consumer_types(alpha, beta, sigma)
        self.plot_type_choices(baseline=False)
        fig.text(0, -0.15, 
            'New parameters: price sensitivity alpha = {}, taste on X mean = {}, variance = {}, range = [{}, {}]'.format(
                self.consumer_alpha, 
                self.consumer_beta, 
                self.consumer_sigma,
                self.consumer_beta-self.consumer_sigma*0.5,
                self.consumer_beta+self.consumer_sigma*0.5), 
            fontsize = 10)

        # set the product_character back
        self.construct_consumer_types(old_par[0], old_par[1], old_par[2])
        
        plt.close()

        return fig



    # I. initialize ----------------------------------------------------
    def construct_product_char(self, delta_X = [0.0,0.0], delta_p=[0.0,0.0]):
        '''give some random number to products 
        delta_p: check effects of price change
        delta_X: check effects of quality change '''

        # quanlity: outside, low quality, high quality
        self.prod_X = np.array([[0, 4.0, 5.0]])
        self.prod_X[0,1] = self.prod_X[0,1] + delta_X[0]
        self.prod_X[0,2] = self.prod_X[0,2] + delta_X[1]

        # price: outside, low quality, high quality
        self.prod_p = np.array([[0, 2.4, 3.5]])
        self.prod_p[0,1] = self.prod_p[0,1]  + delta_p[0]
        self.prod_p[0,2] = self.prod_p[0,2]  + delta_p[1]

        return 


    def construct_consumer_types(self, alpha, beta, sigma):
        '''Consumer types = random taste coeff - social average part
        taste uniform distributed, range = sigma, mean = beta 
        random part = taste - beta '''

        theta = np.linspace(-.5,.5,50)*sigma

        self.consumer_theta = theta
        self.consumer_beta = beta 
        self.consumer_alpha = alpha
        self.consumer_sigma = sigma

        return 

    # II. make choice ----------------------------------------------------
    def consumer_evaluation(self):
        theta = self.consumer_theta.copy()
        beta = self.consumer_beta
        alpha = self.consumer_alpha
        X = self.prod_X.copy()
        p = self.prod_p.copy()

        score = np.exp(np.reshape(theta,(50,1)) @ X + beta*X - alpha * p)
        score_sum = np.sum(score, axis = 1)

        return score, score_sum

    def consumer_choice_withlogiterror(self):
        '''Consumer make choices, with logit error term '''

        (score, score_sum) = self.consumer_evaluation()

        prob = score/np.repeat(np.reshape(score_sum, (50,1)),3,axis = 1)

        return prob
    
    def consumer_choice_purechar(self):
        '''Consumer make choices, without logit error term
        Consumer perfectly predicted by vertical differentiation theory models '''

        (score, score_sum) = self.consumer_evaluation()
        score_max = np.max(score, axis = 1)

        choice = (score == np.repeat(np.reshape(score_max, (50,1)),3,axis = 1)).astype(int)

        return choice


    # III. plot ----------------------------------------------------
    def plot_type_choices(self, baseline = True):

        alpha = self.consumer_alpha
        beta = self.consumer_beta
        sigma = self.consumer_sigma
        theta = self.consumer_theta.copy()

        prob = self.consumer_choice_withlogiterror()
        choice = self.consumer_choice_purechar()

        if baseline:
            self.plot_choices_lines(alpha, beta, sigma, theta, prob, choice)
        else:
            self.plot_addlines(alpha, beta, sigma, theta, prob, choice)

        return 

    def plot_choices_lines(self, alpha, beta, sigma, theta, prob, choice):
        '''Plot types-choice space
        the intuition is also similar to the types-allocation space in mechanism design or screening'''       
        x = np.linspace(0,1,50)
        # subplot 1 choice with logit error
        plt.subplot(1, 2, 1)
        plt.plot(x, prob[:, 0], color = 'silver')
        plt.plot(x, prob[:, 1], color = 'orange')
        plt.plot(x, prob[:, 2], color = 'maroon')
        plt.xlabel('consumer types (random taste)')
        plt.ylabel('prob of choice')
        plt.title('Choice with Logit Error', fontsize = 12)   

        # subplot 2 choice without error term: perfectly predicted by vertical differentiation theory
        plt.subplot(1, 2, 2)
        plt.plot(x, choice[:, 0], color = 'silver', label = 'outside')
        plt.plot(x, choice[:, 1], color = 'orange', label = 'low quality')
        plt.plot(x, choice[:, 2], color = 'maroon', label = 'high quality')
        plt.xlabel('consumer types (random taste)')
        plt.ylabel('prob of choice')
        plt.title('Choice without Error Term (Vertical Diff theory)', fontsize = 12)

        return plt

    def plot_addlines(self, alpha, beta, sigma, theta, prob, choice, save = False, filename = None, figpath = None):

        x = np.linspace(0,1,50)
        # subplot 1 choice with logit error
        #fig = fig_input
        plt.subplot(1, 2, 1)
        plt.plot(x, prob[:, 0], '--', color = 'silver')
        plt.plot(x, prob[:, 1], '--', color = 'orange')
        plt.plot(x, prob[:, 2], '--', color = 'maroon')


        # subplot 2 choice without error term: perfectly predicted by vertical differentiation theory
        plt.subplot(1, 2, 2)
        plt.plot(x, choice[:, 0], '--', color = 'silver')
        plt.plot(x, choice[:, 1], '--', color = 'orange')
        plt.plot(x, choice[:, 2], '--', color = 'maroon')

        return plt









