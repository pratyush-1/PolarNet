import torch
import numpy as np
from reliability_sequence import Reliability_Sequence
from utils import snr_db2sigma,corrupt_signal,min_sum_log_sum_exp,log_sum_exp
class PolarCode:
    def __init__(self, n, K, rs=None, Fr = None, use_cuda = True,infty=1000,hard_decision=False,lse='lse'):
        self.n = n
        self.N =2**n
        self.K = K
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.infty = infty
        self.hard_decision = hard_decision
        self.lse = lse

        if Fr is not None:
            assert len(Fr) == self.N - self.K
            self.frozen_positions = Fr
            self.unsorted_frozen_positions = self.frozen_positions
            self.frozen_positions.sort()

            self.info_positions = np.array(list(set(self.frozen_positions)^set(np.arange(self.N))))
            self.unsorted_info_positions = self.info_positions
            self.info_positions.sort()
        
        else:
            if rs is None:
                # create sequence from good to bad (increasing order of reliability)
                self.reliability_sequence = np.arange(1023,-1,-1)
                #take values < N
                self.rs = self.reliability_sequence[self.reliability_sequence<self.N]
            else:
                self.reliability_sequence = rs
                self.rs = self.reliability_sequence[self.reliability_sequence<self.N]

                assert len(self.rs)==self.N

            # best K bits
            self.info_positions = self.rs[:self.K]
            self.unsorted_info_positions = self.info_positions
            self.info_positions.sort()
            self.unsorted_info_positions=np.flip(self.unsorted_info_positions)
            # worst K bits
            self.frozen_positions = self.rs[self.K:]
            self.unsorted_frozen_positions = self.frozen_positions
            self.frozen_positions.sort()
    
    def encode(self,message):
        u = torch.ones(message.shape[0],self.N,dtype=torch.float).to(message.device) #[B,N]
        u[:,self.info_positions] = message
        for d in range(0,self.n):
            num_bits = 2**d
            for i in np.arange(0,self.N,2*num_bits):
                #[u0 u1] ---> [u0 xor(u0 u1)] #how??
                u = torch.cat((u[:,:i],u[:,i:i+num_bits].clone()*u[:,i+num_bits:i+2*num_bits],u[:,i+num_bits:]),dim=1)
        
        return u
    
    def channel(self,code,snr):
        sigma = snr_db2sigma(snr)

        r = corrupt_signal(code,sigma)

        return r
    def define_partial_arrays(self, llrs):
        # Initialize arrays to store llrs and partial_sums useful to compute the partial successive cancellation process.
        llr_array = torch.zeros(llrs.shape[0], self.n+1, self.N, device=llrs.device)
        llr_array[:, self.n] = llrs
        partial_sums = torch.zeros(llrs.shape[0], self.n+1, self.N, device=llrs.device)
        return llr_array, partial_sums


    def updateLLR(self, leaf_position, llrs, partial_llrs = None, prior = None):

        #START
        depth = self.n
        decoded_bits = partial_llrs[:,0].clone()
        if prior is None:
            prior = torch.zeros(self.N) #priors
        llrs, partial_llrs, decoded_bits = self.partial_decode(llrs, partial_llrs, depth, 0, leaf_position, prior, decoded_bits)
        return llrs, decoded_bits


    def partial_decode(self, llrs, partial_llrs, depth, bit_position, leaf_position, prior, decoded_bits=None):
        # Function to call recursively, for partial SC decoder.
        # We are assuming that u_0, u_1, .... , u_{leaf_position -1} bits are known.
        # Partial sums computes the sums got through Plotkin encoding operations of known bits, to avoid recomputation.
        # this function is implemented for rate 1 (not accounting for frozen bits in polar SC decoding)

        # print("DEPTH = {}, bit_position = {}".format(depth, bit_position))
        half_index = 2 ** (depth - 1)
        leaf_position_at_depth = leaf_position // 2**(depth-1) # will tell us whether left_child or right_child

        # n = 2 tree case
        if depth == 1:
            # Left child
            left_bit_position = 2*bit_position
            if leaf_position_at_depth > left_bit_position:
                u_hat = partial_llrs[:, depth-1, left_bit_position:left_bit_position+1]
            elif leaf_position_at_depth == left_bit_position:
                if self.lse == 'minsum':
                    Lu = min_sum_log_sum_exp(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]).sum(dim=1, keepdim=True)
                elif self.lse == 'lse':
                    Lu = log_sum_exp(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]).sum(dim=1, keepdim=True)
                # Lu = log_sum_avoid_zero_NaN(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]).sum(dim=1, keepdim=True)
                #print(Lu.device, prior.device, torch.ones_like(Lu).device)
                llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index] = Lu + prior[left_bit_position]*torch.ones_like(Lu)
                if self.hard_decision:
                    u_hat = torch.sign(Lu)
                else:
                    u_hat = torch.tanh(Lu/2)

                decoded_bits[:, left_bit_position] = u_hat.squeeze(1)

                return llrs, partial_llrs, decoded_bits

            # Right child
            right_bit_position = 2*bit_position + 1
            if leaf_position_at_depth > right_bit_position:
                pass
            elif leaf_position_at_depth == right_bit_position:
                Lv = u_hat * llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index] + llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]
                llrs[:, depth-1, right_bit_position*half_index:(right_bit_position+1)*half_index] = Lv + prior[right_bit_position] * torch.ones_like(Lv)
                if self.hard_decision:
                    v_hat = torch.sign(Lv)
                else:
                    v_hat = torch.tanh(Lv/2)
                decoded_bits[:, right_bit_position] = v_hat.squeeze(1)
                return llrs, partial_llrs, decoded_bits

        # General case
        else:
            # LEFT CHILD
            # Find likelihood of (u xor v) xor (v) = u
            # Lu = log_sum_exp(torch.cat([llrs[:, :half_index].unsqueeze(2), llrs[:, half_index:].unsqueeze(2)], dim=2).permute(0, 2, 1))
            
            left_bit_position = 2*bit_position
            if leaf_position_at_depth > left_bit_position:
                Lu = llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index]
                u_hat = partial_llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index]
            else:
                if self.lse == 'minsum':
                    Lu = min_sum_log_sum_exp(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index])
                elif self.lse == 'lse':
                    # Lu = log_sum_avoid_zero_NaN(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index])
                    Lu = log_sum_exp(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index])

                llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index] = Lu
                llrs, partial_llrs, decoded_bits = self.partial_decode(llrs, partial_llrs, depth-1, left_bit_position, leaf_position, prior, decoded_bits)

                return llrs, partial_llrs, decoded_bits

            # RIGHT CHILD
            right_bit_position = 2*bit_position + 1

            Lv = u_hat * llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index] + llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]
            llrs[:, depth-1, right_bit_position*half_index:(right_bit_position+1)*half_index] = Lv
            llrs, partial_llrs, decoded_bits = self.partial_decode(llrs, partial_llrs, depth-1, right_bit_position, leaf_position, prior, decoded_bits)

            return llrs, partial_llrs, decoded_bits

    def updatePartialSums(self, leaf_position, decoded_bits, partial_llrs):

        u = decoded_bits.clone()
        u[:, leaf_position+1:] = 0

        for d in range(0, self.n):
            partial_llrs[:, d] = u
            num_bits = 2**d
            for i in np.arange(0, self.N, 2*num_bits):
                # [u v] encoded to [u xor(u,v)]
                u = torch.cat((u[:, :i], u[:, i:i+num_bits].clone() * u[:, i+num_bits: i+2*num_bits], u[:, i+num_bits:]), dim=1)
        partial_llrs[:, self.n] = u
        return partial_llrs

    def sc_decode_new(self, corrupted_codewords, snr, use_gt = None, channel = 'awgn'):

        assert channel in ['awgn', 'bsc']

        if channel == 'awgn':
            noise_sigma = snr_db2sigma(snr)
            llrs = (2/noise_sigma**2)*corrupted_codewords
            
        # step-wise implementation using updateLLR and updatePartialSums

        priors = torch.zeros(self.N)
        priors[self.frozen_positions] = self.infty

        u_hat = torch.zeros(corrupted_codewords.shape[0], self.N, device=corrupted_codewords.device)
        llr_array, partial_llrs = self.define_partial_arrays(llrs)
        for ii in range(self.N):
            #start = time.time()
            llr_array , decoded_bits = self.updateLLR(ii, llr_array.clone(), partial_llrs, priors)
            #print('SC update : {}'.format(time.time() - start), corrupted_codewords.shape[0])
            if use_gt is None:
                u_hat[:, ii] = torch.sign(llr_array[:, 0, ii])
            else:
                u_hat[:, ii] = use_gt[:, ii]
            #start = time.time()
            partial_llrs = self.updatePartialSums(ii, u_hat, partial_llrs)
            #print('SC partial: {}s, {}', time.time() - start, 'frozen' if ii in self.frozen_positions else 'info')
        decoded_bits = u_hat[:, self.info_positions]
        return llr_array[:, 0, :].clone(), decoded_bits
        
def get_frozen(N,K,rs):
    rs = rs[rs<N]
    Fr = rs[K:].copy()
    Fr.sort()
    return Fr