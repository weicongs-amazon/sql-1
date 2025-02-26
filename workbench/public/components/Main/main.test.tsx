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
 *   Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import React from "react";
import "@testing-library/jest-dom/extend-expect";
import { render, fireEvent } from "@testing-library/react";
import { httpClientMock } from "../../../test/mocks";
import {
  mockQueryResultJDBCResponse,
  mockNotOkQueryResultResponse,
  mockQueryTranslationResponse,
  mockResultWithNull
} from "../../../test/mocks/mockData";
import Main from "./main";

describe("<Main /> spec", () => {

  it("renders the component", () => {
    render(
      <Main httpClient={httpClientMock} />
    );
    expect(document.body.children[0]).toMatchSnapshot();
  });

  it("click run button, and response is ok", async () => {
    const client = httpClientMock;
    client.post = jest.fn().mockResolvedValue(mockQueryResultJDBCResponse);

    const { getByText } = render(
      <Main httpClient={client} />
    );
    const onRunButton = getByText('Run');
    const asyncTest = () => {
      fireEvent.click(onRunButton);
    };
    await asyncTest();
    expect(document.body.children[0]).toMatchSnapshot();
  });

  it("click run button, response fills null and missing values", async () => {
    const client = httpClientMock;
    client.post = jest.fn().mockResolvedValue(mockResultWithNull);

    const { getByText } = render(
      <Main httpClient={client} />
    );
    const onRunButton = getByText('Run');
    const asyncTest = () => {
      fireEvent.click(onRunButton);
    };
    await asyncTest();
    expect(document.body.children[0]).toMatchSnapshot();
  })

  it("click run button, and response causes an error", async () => {
    const client = httpClientMock;
    client.post = jest.fn().mockRejectedValue('err');

    const { getByText } = render(
      <Main httpClient={client} />
    );
    const onRunButton = getByText('Run');
    const asyncTest = () => {
      fireEvent.click(onRunButton);
    };
    await asyncTest();
    expect(document.body.children[0]).toMatchSnapshot();
  });

  it("click run button, and response is not ok", async () => {
    const client = httpClientMock;
    client.post = jest.fn().mockResolvedValue(mockNotOkQueryResultResponse);

    const { getByText } = render(
      <Main httpClient={client} />
    );
    const onRunButton = getByText('Run');
    const asyncTest = () => {
      fireEvent.click(onRunButton);
    };
    await asyncTest();
    expect(document.body.children[0]).toMatchSnapshot();
  });

  it("click translation button, and response is ok", async () => {
    const client = httpClientMock;
    client.post = jest.fn().mockResolvedValue(mockQueryTranslationResponse);
    const { getByText } = render(
      <Main httpClient={client} />
    );
    const onTranslateButton = getByText('Explain');
    const asyncTest = () => {
      fireEvent.click(onTranslateButton);
    };
    await asyncTest();
    expect(document.body.children[0]).toMatchSnapshot();
  });

  it("click clear button", async () => {
    const client = httpClientMock;
    const { getByText } = render(
      <Main httpClient={client} />
    );
    const onClearButton = getByText('Clear');
    const asyncTest = () => {
      fireEvent.click(onClearButton);
    };
    await asyncTest();
    expect(client.post).not.toHaveBeenCalled();
    expect(document.body.children[0]).toMatchSnapshot();
  });
});
