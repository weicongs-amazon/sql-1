/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

/*
 * Copyright <2019> Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * A copy of the License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * or in the "license" file accompanying this file. This file is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 *
 */

package com.amazon.opendistroforelasticsearch.jdbc.types;

import java.sql.SQLException;
import java.util.Map;

public class BooleanType implements TypeHelper<Boolean> {

    public static final BooleanType INSTANCE = new BooleanType();

    private BooleanType() {

    }
    
    @Override
    public Boolean fromValue(Object value, Map<String, Object> conversionParams) throws SQLException {
        if (value == null) {
            return false;
        }

        if (value instanceof Boolean) {
            return (Boolean) value;
        } else if (value instanceof String) {
            return asBoolean((String) value);
        } else {
            throw objectConversionException(value);
        }
    }

    private Boolean asBoolean(String value) throws SQLException {
        try {
            return Boolean.valueOf(value);
        } catch (NumberFormatException nfe) {
            throw stringConversionException(value, nfe);
        }
    }


    @Override
    public String getTypeName() {
        return "Boolean";
    }
}
