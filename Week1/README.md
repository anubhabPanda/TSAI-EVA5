1.What are Channels and Kernels (according to EVA)?
=================================================
  * Channel is a collection of contextually similar imformation placed together. Another interpretation of channel is a collection of similar or same feature. Sometime, a channel is also referred to as a feature map. Few examples of channels are -
    
    * All alphanumeric characters can be considered as separate channels (A-Z, a-z, 0-9, punctuations, other special characters, etc).
    * Different parts of speech can be considered as separate channels (Nouns can be considered as one channel, verbs as another and so on).
    * Mammals can be considered as a single channel in Animal Kingdom.
    * Each note in a song can be thought of as a different channel.
    * In a painting, all the shades of red can be considered a channel.

  * Kernel or Filter is a feature extractor. A kernel is a matrix of randomly initialized numbers whose output is a specific feature such as vertical edge or horizontal edge, etc. The numbers in the matrix are called weights which are learned during backpropagation. Usually a 3 X 3 kernel is used. 
    
