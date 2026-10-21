# Dataset Documentation: Ames Housing Dataset

## 1. Overview and Primary Source
- **Dataset Name**: Ames Housing Dataset
- **Official Competition**: [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
- **Original Compilation**: Compiled by Professor Dean De Cock (Truman State University) for use in data science education.
- **Geographic Context**: Ames, Iowa, United States.
- **Records (Training Data)**: 1,460 residential property transactions.
- **Features**: 79 explanatory variables describing almost every aspect of residential homes, plus identifier (`Id`) and target (`SalePrice`).

## 2. Key Attribute Categories
- **Target Variable**:
  - `SalePrice`: Continuous variable representing property sale price in US Dollars (USD).
- **Basic Property Characteristics**:
  - `GrLivArea`: Above-grade (ground living) area square footage.
  - `LotArea`: Lot size in square feet.
  - `BedroomAbvGr`: Bedrooms above ground level (excludes basement bedrooms).
  - `FullBath` & `HalfBath`: Full and half bathrooms above grade.
  - `OverallQual`: Overall material and finish rating (1 = Very Poor, 10 = Very Excellent).
  - `OverallCond`: Overall condition rating (1 = Very Poor, 10 = Very Excellent).
  - `GarageCars` & `GarageArea`: Car capacity and size of garage in square feet.
  - `TotalBsmtSF`: Total square feet of basement area.
  - `HouseStyle`: Style of dwelling (e.g., 1Story, 2Story, Split-Level).
  - `KitchenQual`: Kitchen quality (Ex, Gd, TA, Fa).
  - `ExterQual`: Exterior material quality (Ex, Gd, TA, Fa).
- **Age and Renovation Attributes**:
  - `YearBuilt`: Original construction year (range: 1872 to 2010).
  - `YearRemodAdd`: Remodel date (same as construction date if no remodeling or addition).
  - `YrSold`: Year property was sold (2006 to 2010).
  - *Derived Age*: `PropertyAge = YrSold - YearBuilt`.
  - *Derived Remodel Age*: `YearsSinceRemodel = YrSold - YearRemodAdd`.
- **Location Attributes**:
  - `Neighborhood`: 25 distinct physical locations within Ames city limits (e.g., North Ames, College Creek, Old Town, Edwards, Somerset, Gilbert, Northridge Heights).
  - *Note*: No fabricated city-center distance is introduced; genuine neighborhood categories are preserved.
- **Road Access and Proximity**:
  - `Street`: Type of road access to property (`Pave`, `Grvl`).
  - `Alley`: Type of alley access (`Grvl`, `Pave`, `NA` for no alley access).
  - `Condition1` & `Condition2`: Proximity to various conditions (e.g., arterial streets `Artery`, feeder streets `Feedr`, railroads `RRA/RRNe`, positive parks `PosN/PosA`).
- **View and Topography**:
  - `LandContour`: Flatness of property (`Lvl` = Near Flat/Level, `Bnk` = Banked, `HLS` = Hillside, `Low` = Depression).
  - `LotConfig`: Lot configuration (`Inside`, `Corner`, `CulDSac`, `FR2`, `FR3`).

## 3. Important Boundaries and Academic Limitations
1. **Currency and Geographic Applicability**:
   - The dataset represents real estate dynamics of Ames, Iowa (US) between 2006 and 2010.
   - Predictions are expressed in **USD ($)** and cannot be treated as valid valuations for Indian properties (INR), which depend on entirely different local factors (e.g., RERA approvals, metro connectivity, circle rates, floor rise index).
2. **Absence of Coastline / Sea-View**:
   - Ames is an inland city in Iowa. There are **no ocean or sea-view attributes** in this dataset. We honestly document this limitation rather than fabricating synthetic ocean-view values.
3. **Constructed Residential Properties vs. Vacant Land**:
   - All 1,460 records represent constructed single-family or attached residential dwellings with foundations, utilities, and structures.
   - Vacant agricultural or commercial plots are **not** present in this dataset. Plot prediction is therefore deferred to the future scope.

## 4. Fallback Support: California Housing
If the Ames Housing dataset is inaccessible in a restricted environment, the codebase includes a documented fallback using Scikit-Learn's `fetch_california_housing()`. However:
- California housing evaluates 1990 census block groups, not individual homes.
- The target is median house value in units of $100,000.
- Metrics from California Housing must **never** be mixed into the Ames Housing evaluation table.
