"""
    A tool for helping keep track of different program states.
"""

import collections
import pickle

FEED = 0
DATA = 1

class ContextManager:
    """
    This tool can simplify program in a combinatoric state space.
    
    During instantiation, you must declare what kind of state data this instance
    will be managing.  This tool was written explicitly for managing TensorFlow
    feed dictionaries, but it can also be used to manage other kinds of data.
    The two parameters to the constructor control what kind of state you are 
    looking to manage.  At least one option must be chosen, and choosing both is
    also valid.  This choice cannot be changed after instantiation.
    
    If force_substrate_coherence is True (default), all of the substate data in
    each root context must have the same structure (but different root contexts
    may have different structure.)
    
    Some program stateful information is discrete, and this type of data is stored in
    contexts.  A context contains a set of sub-states or which one is always active.
    But other program stateful information is not discrete, and so the 
    context/sub-state architecture doesn't really make sense.  For this type of data,
    I have included two public attributes: self.default_feeds, and self.data.  
    These are both simply dicts with no special functions or attributes.  You can store
    whatever you need in these, both as a container for program parameters and also
    as an easy method of storing and retrieving values in files, via the 
    ContextManager's save and load functions.  They should be read/write safe.
    The contents in self.data will be always returned (with possible override) 
    when calling get_data(), but not get_data_from_context(), and the same is true for
    default_feeds.  Note that in order to be able to use either of these, the 
    corresponding mode has to be set.
    
    Note that whatever goes into data has to be picklable, so no tf ops.  But
    of course the feeds should only be tf.placeholder
    
    """
    def __init__(
        self,
        use_feeds, 
        use_data, 
        force_substate_coherence=True
    ):
        if use_feeds == False and use_data == False:
            raise RuntimeError("ContextManager construction: "
                "at least one of uses_feeds or uses_data be true"
            )
        
        self._uses_feeds = use_feeds
        self._uses_data = use_data
        self._force_substate_coherence = force_substate_coherence
        
        if self._uses_data:
            self._root_data_contexts = {}
            self.data = {}
        if self._uses_feeds:
            self._root_feed_contexts = {}
            self.default_feeds = {}
        self._active_states = {}
        
    def add_context(self, context, substates, active_value):
        """
        Adds a new root context.
        
        A root context is a container of one dimension of possible program states.
        Exactly one of these states will always be active.  Multiple root contexts
        can be defined.  All are independant and each root context always has an
        active state.  The total state space of the program is the combination of
        all valid_values of all root contexts.
        
        Parameters
        ----------
        context : hashable
            A value that will be used as a key in a dict that indexes each class
            of states handled by the context manager.  This is a type of state
            and not an individual state itself.  Should probably be a string, for 
            clarity, but this isn't a requirement.
            
        substates : dict
            A dict containing substate, value pairs.  The substate is a key to
            individual instances of the program state that exist within this root
            context.  The value for each key will contain information about the 
            substate itself.  
            
            Valid values depend on how the ContextManager was initialized.  If 
            uses_feeds was set to True and uses_data was set to False, the values must 
            be a dict that contains the TensorFlow feed dict to use when each 
            subcontext is active.  If uses_feeds was set to False and uses_data was 
            set to True then this must be a dict that contains the data you want 
            to store in each subcontext.  If both were set to True then the value must 
            be a tuple containing both, with the feed dict first and the data second.
            
        active_value : hashable
            Every root context must always have one of its states active.  This 
            parameter tells which subcontext state should be active from the start.
            It must be one of the keys in valid_values.
            
        """
        
        # Error check context.  Then add it to the root_contexts
        try:
            if self._uses_data:
                self._root_data_contexts[context] = {}
            if self._uses_feeds:
                self._root_feed_contexts[context] = {}
        except TypeError as e:
            raise TypeError(
                "ContextManager.add_context: parameter context not hashable."
            ) from e
            
        # Add each substate to the root context
        if type(substates) is not dict:
            raise TypeError(
                "ContextManager.add_context: parameter substates must be a dict."
            )
        else:
            for key, value in substates.items():
                self.add_state_to_context(context, key, value)
            
        # Activate the active value
        try:
            self.set_state(context, active_value)
        except TypeError as e:
            raise TypeError(
                "ContextManager.add_context: active_value not hashable."
            ) from e
        
    def set_state(self, context, sub_state):
        """
            Sets the given root context to the given substate
        """
        try:
            vs = self.get_valid_states_from_context(context)
        except KeyError as e:
            raise ValueError(
                f"ContextManager.set_state: context {context} is not "
                f"a valid context."
            ) from e
            
        if sub_state in vs:
            self._active_states[context] = sub_state
        else:
            raise ValueError(
                f"ContextManager.set_state: sub-state {sub_state} not a valid. "
                f"sub-state for root context {context}.  Call add_state to add a "
                f"new sub-state to a context."
            )
        if self._uses_feeds:
            self.rebuild_feeds()
        if self._uses_data:
            self.rebuild_data()
    
    def get_active_state(self, context):
        try:
            return self._active_states[context]
        except KeyError as e:
            raise KeyError(
                f"ContextManager.get_valid_states_from_context: context {context} "
                f"is not a valid context for this instance."
            ) from e
            
    def add_state_to_context(self, context, sub_state, data):
        """
            Adds a new sub_state to an already existing context.
        """
        try:
            if sub_state in self.get_valid_contexts():
                raise ValueError(
                    f"ContextManager.add_state_to_context: sub-state {sub_state} "
                    f"already exists in context {context}."
                )
        except KeyError as e:
            raise ValueError(
                f"ContextManager.add_state_to_context: context {context} is not "
                f"already in this context.  Use add_context instead to add a new ."
                f"context."
            ) from e
        
        # check that data is the correct shape and type    
        if self._uses_feeds and self._uses_data:
            if type(data) is not tuple:
                raise ValueError(
                    f"ContextManager.add_context: substate {sub_state} must "
                    f"index a 2-tuple."
                )
            if len(data) != 2:
                raise ValueError(
                    f"ContextManager.add_context: substate {sub_state} must "
                    f"index a 2-tuple."
                )
            if (type(data[0]) is not dict) or (type(data[1]) is not dict):
                raise ValueError(
                    f"ContextManager.add_context: substate {sub_state} must "
                    f"index a 2-tuple of dicts."
                )
        else: # not using both modes, so expect just a single dict
            if type(data) is not dict:
                raise ValueError(
                    f"ContextManager.add_context: substate {sub_state} must "
                    f"index a dict."
                )
        
        if self._force_substate_coherence:
            # get any other substate (if they exist) and compare its keys to
            # the new substate's keys
            try:
                if self._uses_data:
                    any_other_data = next(iter(
                        self._root_data_contexts[context].values()
                    ))
                if self._uses_feeds:
                    any_other_feed = next(iter(
                        self._root_feed_contexts[context].values()
                    ))
                if self._uses_data and self._uses_feeds:
                    if (
                        set(any_other_data.keys()) != set(data[DATA].keys()) or
                        set(any_other_feed.keys()) != set(data[FEED].keys())
                    ):
                        raise ValueError(
                            f"ContextManager.add_context: set to force "
                            f"sub-state coherence, but found a data "
                            f"inconsistency: sub-state {sub_state} does not have "
                            f"the same entries as the previous sub-state."
                        )
                else:
                    if self._uses_data:
                        any_other_keys = set(any_other_data.keys())
                    else:
                        any_other_keys = set(any_other_feed.keys())
                    if any_other_keys != set(data.keys()):
                        raise ValueError(
                            f"ContextManager.add_context: set to force "
                            f"sub-state coherence, but found a data "
                            f"inconsistency: sub-state {sub_state} does not have "
                            f"the same entries as the previous sub-state."
                        )
            except StopIteration:
                # If this fails, the context has no substates, so there is nothing
                # to enforce coherence with.
                pass
        
        # actually add the state
        if self._uses_data and self._uses_feeds:
            self._root_data_contexts[context][sub_state] = data[DATA]
            self._root_feed_contexts[context][sub_state] = data[FEED]
        else:
            if self._uses_data:
                self._root_data_contexts[context][sub_state] = data
            else:
                self._root_feed_contexts[context][sub_state] = data
        
    def update_data_in_context(self, context, new_data):
        """
            Updates the data stored in every state of a given root context, if force
            data coherence is true.  Otherwise may only update the data stored in some
            of the states.  May both replace existing data values and add new entries
            to the data dict, but does not delete any data catagories.
        """
        if self._uses_data:
            try:
                old_sub_states = set(self.get_valid_states_from_context(context))
                if self._force_substate_coherence:
                    if old_sub_states != set(new_data.keys()):
                        raise ValueError(
                            f"ContextManager.update_data_in_context: forcing data "
                            f"coherence, but new_data does not have the same "
                            f"sub-states as currently exist in the given context."
                        )
                for substate in self.get_valid_states_from_context(context):
                    self._root_data_contexts[context][substate].update(
                        new_data[substate]
                    )
                self.rebuild_data()
            except KeyError as e:
                raise KeyError(
                    f"ContextManager.update_data_in_context: context {context} "
                    f"is not a valid context for this instance."
                ) from e
        else:
            raise RuntimeError(
                f"ContextManager.update_data_in_context was called, but this "
                f"context manager is not in data mode."
            )
            
    def update_feeds_in_context(self, context, new_feeds):
        """
            Updates the feeds stored in every state of a given root context, if force
            data coherence is true.  Otherwise may only update the feeds stored in some
            of the states.  May both replace existing feed values and add new entries
            to the feed dict, but does not delete any feed catagories.
        """
        if self._uses_feeds:
            try:
                old_sub_states = set(self.get_valid_states_from_context(context))
                if self._force_substate_coherence:
                    if old_sub_states != set(new_feeds.keys()):
                        raise ValueError(
                            f"ContextManager.update_feeds_in_context: forcing data "
                            f"coherence, but new_feeds does not have the same "
                            f"sub-states as currently exist in the given context."
                        )
                for substate in self.get_valid_states_from_context(context):
                    self._root_feed_contexts[context][substate].update(
                        new_feeds[substate]
                    )
                self.rebuild_feeds()
            except KeyError as e:
                raise KeyError(
                    f"ContextManager.update_feeds_in_context: context {context} "
                    f"is not a valid context for this instance."
                ) from e
        else:
            raise RuntimeError(
                f"ContextManager.update_feeds_in_context was called, but this "
                f"context manager is not in feed mode."
            )
            
    def get_data_from_context(self, context):
        """
            Gets the data dict from whatever substate context is currently in, but
            only if in uses_data mode.
        """
        if self._uses_data:
            return self._root_data_contexts[context][self._active_states[context]]
        else:
            raise RuntimeError(
                f"ContextManager.get_data was called, but this context manager is "
                f"not in data mode."
            )
            
    def get_feeds_from_context(self, context):
        """
            Gets the feed dict from whatever substate context is currently in, but
            only if in uses_feeds mode.
        """
        if self._uses_feeds:
            return self._root_feed_contexts[context][self._active_states[context]]
        else:
            raise RuntimeError(
                f"ContextManager.get_data was called, but this context manager is "
                f"not in feed mode."
            )
            
    def rebuild_data(self):
        """
            Updates the internal data state dict, which is used so that we don't have
            to recompute the output every time get_data is called.
        """
        self._all_data = {}
        self._all_data.update(self.data)
        for context in self._active_states.keys():
            self._all_data.update(self.get_data_from_context(context))
            
    def get_data(self):
        """
            Gets the data dict from the active state across all contexts, packed into 
            a single dict.
        """
        if self._uses_data:
            try:
                return self._all_data
            except AttributeError:
                self.rebuild_data()
                return self._all_data
        else:
            raise RuntimeError(
                f"ContextManager.get_data was called, but this context manager is "
                f"not in data mode."
            )
    
    def rebuild_feeds(self):
        """
            Updates the internal feed state dict, which is used so that we don't have
            to recompute the output every time get_feeds is called.
        """
        self._all_feeds = {}
        self._all_feeds.update(self.default_feeds)
        for context in self._active_states.keys():
            self._all_feeds.update(self.get_feeds_from_context(context))
            
    def get_feeds(self, overrides={}):
        """
            Gets the feed dict from the active state across all contexts, packed into 
            a single dict.
            
            This function is designed to be plugged directly into a TensorFlow
            session.run call, as: session.run(tensor, feed_dict=context.get_feeds())
            If you pass this function a dict (of other valid feeds), those values
            will merge with and possibly override the values generated by this 
            context.  This can both add new feed values that are not handled by this 
            context, and can manually override one or more of the feeds handled by 
            this context.  Of course if you want to do the latter, why not just change 
            the state?
        """
        if self._uses_feeds:
            output = {}
            try:
                output.update(self._all_feeds)
            except AttributeError:
                self.rebuild_feeds()
                output.update(self._all_feeds)
            output.update(overrides)
            return output
        else:
            raise RuntimeError(
                f"ContextManager.get_data was called, but this context manager is "
                f"not in feed mode."
            )
            
    def get_valid_contexts(self):
        """
            Get a list of keys for all contexts.
        """
        if self._uses_data:
            return self._root_data_contexts.keys()
        else:
            return self._root_feed_contexts.keys()
        
    def get_valid_states_from_context(self, context):
        """
            Get a list of keys for the states within a context
        """
        try:
            if self._uses_data:
                return self._root_data_contexts[context].keys()
            else:
                return self._root_feed_contexts[context].keys()
        except KeyError as e:
            raise KeyError(
                f"ContextManager.get_valid_states_from_context: context {context} "
                f"is not a valid context for this instance."
            ) from e
            
    def save(self, filename):
        """
            Pickle this class manager instance into a file.
            
            Placeholders cannot be pickled, and since they belong to a tf graph, this
            would be a problem anyway.  So we have to replace each placeholder (a key
            in the feed dicts) with its name (a string).  Fortunately names are unique,
            so we can restore them in load.  But unfortunately, this means making
            a copy of the context manager.  And I can't use the python copy tools,
            because tf forbids it.
            
        """
        
        def purge_placeholders(feed_dict):
            output = {}
            for key, value in feed_dict.items():
                output[key.name] = value
            return output
            
        def deeper_purge_placeholders(root_contexts):
            new_root = {}
            for orig_root, orig_states in root_contexts.items():
                new_states = {}
                for substate, orig_feeds in orig_states.items():
                    new_states[substate] = purge_placeholders(orig_feeds)
                new_root[orig_root] = new_states
            return new_root
        
        copy_cm = ContextManager(
            self._uses_feeds,
            self._uses_data,
            self._force_substate_coherence
        )
        
        if self._uses_data:
            copy_cm._root_data_contexts = {}
            copy_cm._root_data_contexts.update(self._root_data_contexts)
            copy_cm.data = {}
            copy_cm.data.update(self.data)
        if self._uses_feeds:
            copy_cm._root_feed_contexts = deeper_purge_placeholders(
                self._root_feed_contexts
            )
            copy_cm.default_feeds = purge_placeholders(self.default_feeds)
            
        copy_cm._active_states = self._active_states
        
        with open(filename, 'wb') as outFile:
            pickle.dump(copy_cm, outFile, pickle.HIGHEST_PROTOCOL)
            
def load(filename, placeholder_list, verify_context_manager=True):
    """
        Create a new ContextManager instance by loading from a file.
    """
    with open(filename, 'rb') as inFile:
        obj = pickle.load(inFile)
        if verify_context_manager:
            assert isinstance(obj, ContextManager), ("Error: attempted to load"
                "a context manager from a file that did not actually contain a "
                "context manager."
            )
    placeholder_names = {p.name: p for p in placeholder_list}
    
    def restore_placeholders(feed_dict, placeholder_names):
        output = {}
        for key, value in feed_dict.items():
            output[placeholder_names[key]] = value
        return output
        
    def deeper_restore_placeholders(root_contexts, placeholder_names):
        new_root = {}
        for orig_root, orig_states in root_contexts.items():
            new_states = {}
            for substate, orig_feeds in orig_states.items():
                new_states[substate] = restore_placeholders(
                    orig_feeds,
                    placeholder_names
                )
            new_root[orig_root] = new_states
        return new_root
    
    obj.default_feeds = restore_placeholders(obj.default_feeds, placeholder_names)
    obj._root_feed_contexts = deeper_restore_placeholders(
        obj._root_feed_contexts,
        placeholder_names
    )
    return obj
        
            
            
            
            
            
            
            
            
            
            
