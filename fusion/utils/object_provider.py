class Factory:
	def __init__(self):
		self._objects = {}

	def register_object(self, key, model):
		"""

		:param key:
		:param model:
		:return:
		"""
		self._objects[key] = model

	def create(self, key, **kwargs):
		"""

		:param key:
		:param kwargs:
		:return:
		"""
		obj = self._objects.get(key)
		if not obj:
			raise ValueError(key)
		return obj(**kwargs)


class ObjectProvider(Factory):
	def get(self, idx, **kwargs):
		return self.create(idx, **kwargs)
