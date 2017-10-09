class Space(object):
  """Defines the observation and action spaces, so you can write generic
  code that applies to any Env. For example, you can choose a random
  action.
  """

  def sample(self):
    """
    Uniformly randomly sample a random element of this space
    """
    raise NotImplementedError

  def contains(self, x):
    """
    Return boolean specifying if x is a valid
    member of this space
    """
    raise NotImplementedError

  def to_jsonable(self, sample_n):
    """Convert a batch of samples from this space to a JSONable data type."""
    # By default, assume identity is JSONable
    return sample_n

  def from_jsonable(self, sample_n):
    """Convert a JSONable data type to a batch of samples from this space."""
    # By default, assume identity is JSONable
    return sample_n