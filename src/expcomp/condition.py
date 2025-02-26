from typing import Any, Callable, Optional
import pandas as pd


class Condition:
    """
    A class to filter pandas DataFrames based on specified conditions.
    
    This class provides a flexible way to apply filtering conditions to
    experiment data with support for standard operators and custom functions.
    """
    
    def __init__(self, key: str, operator: str, value: Any, custom_fn: Optional[Callable] = None):
        """
        Initialize a condition for filtering DataFrame.
        
        Args:
            key: The column name to apply the condition on
            operator: The operator to use for comparison ('==', '!=', '>', '<', '>=', '<=', 'contains', 'custom')
            value: The value to compare against
            custom_fn: Optional custom function for filtering when operator='custom'
        """
        self.key = key
        self.operator = operator
        self.value = value
        self.custom_fn = custom_fn
        
        # Validate that custom_fn is provided when operator is 'custom'
        if self.operator == 'custom' and self.custom_fn is None:
            raise ValueError("Custom function must be provided when operator is 'custom'")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the condition to filter the DataFrame.
        
        Args:
            df: The pandas DataFrame to filter
            
        Returns:
            A filtered pandas DataFrame
            
        Raises:
            ValueError: If an unsupported operator is provided
            KeyError: If the specified key does not exist in the DataFrame
        """
        # Verify the key exists in the DataFrame
        if self.key not in df.columns and self.operator != 'custom':
            raise KeyError(f"Column '{self.key}' not found in DataFrame")
            
        if self.operator == '==':
            return df[df[self.key] == self.value]
        elif self.operator == '!=':
            return df[df[self.key] != self.value]
        elif self.operator == '>':
            return df[df[self.key] > self.value]
        elif self.operator == '<':
            return df[df[self.key] < self.value]
        elif self.operator == '>=':
            return df[df[self.key] >= self.value]
        elif self.operator == '<=':
            return df[df[self.key] <= self.value]
        elif self.operator == 'contains':
            # Handle non-string columns gracefully
            if not pd.api.types.is_string_dtype(df[self.key]):
                df_str = df[self.key].astype(str)
                return df[df_str.str.contains(str(self.value), na=False)]
            return df[df[self.key].str.contains(self.value, na=False)]
        elif self.operator == 'in':
            return df[df[self.key].isin(self.value)]
        elif self.operator == 'not in':
            return df[~df[self.key].isin(self.value)]
        elif self.operator == 'isna':
            return df[df[self.key].isna()]
        elif self.operator == 'notna':
            return df[df[self.key].notna()]
        elif self.operator == 'custom':
            # Apply custom function for filtering
            return self.custom_fn(df, self.key, self.value)
        else:
            raise ValueError(f"Unsupported operator: {self.operator}")
    
