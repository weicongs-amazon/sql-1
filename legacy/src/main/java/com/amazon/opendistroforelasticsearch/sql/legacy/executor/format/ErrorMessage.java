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
 *   Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License").
 *   You may not use this file except in compliance with the License.
 *   A copy of the License is located at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   or in the "license" file accompanying this file. This file is distributed
 *   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *   express or implied. See the License for the specific language governing
 *   permissions and limitations under the License.
 */

package com.amazon.opendistroforelasticsearch.sql.legacy.executor.format;

import org.opensearch.rest.RestStatus;
import org.json.JSONObject;

public class ErrorMessage<E extends Exception> {

    protected E exception;

    private int status;
    private String type;
    private String reason;
    private String details;

    public ErrorMessage(E exception, int status) {
        this.exception = exception;
        this.status = status;

        this.type = fetchType();
        this.reason = fetchReason();
        this.details = fetchDetails();
    }

    private String fetchType() {
        return exception.getClass().getSimpleName();
    }

    protected String fetchReason() {
        return status == RestStatus.BAD_REQUEST.getStatus()
                ? "Invalid SQL query"
                : "There was internal problem at backend";
    }

    protected String fetchDetails() {
        // Some exception prints internal information (full class name) which is security concern
        //return exception.toString();
        return emptyStringIfNull(exception.getLocalizedMessage());
    }

    private String emptyStringIfNull(String str) {
        return str != null ? str : "";
    }

    @Override
    public String toString() {
        JSONObject output = new JSONObject();

        output.put("status", status);
        output.put("error", getErrorAsJson());

        return output.toString(2);
    }

    private JSONObject getErrorAsJson() {
        JSONObject errorJson = new JSONObject();

        errorJson.put("type", type);
        errorJson.put("reason", reason);
        errorJson.put("details", details);

        return errorJson;
    }
}
