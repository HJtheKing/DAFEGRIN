package com.chj.dafegrin.domain.bert.service;

import com.chj.dafegrin.domain.bert.dto.BertRequestDTO;
import com.chj.dafegrin.domain.bert.dto.BertResponseDTO;
import com.chj.dafegrin.domain.bert.dto.BertResult;
import com.chj.dafegrin.domain.dailyindex.repository.DailyIndexRepository;
import com.chj.dafegrin.domain.news.repository.NewsRepository;
import com.chj.dafegrin.domain.news.service.NewsService;
import com.chj.dafegrin.entity.DailyIndex;
import com.chj.dafegrin.entity.News;
import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.time.LocalDate;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
@Transactional
@RequiredArgsConstructor
@Slf4j
public class BertService {

    private final NewsService newsService;
    private final NewsRepository newsRepository;
    private final DailyIndexRepository dailyIndexRepository;

    public void getBert(String keyword, String url) {
        /**
         * news를 개별 posProb을 가진 객체 리스트로 변환해야됨
         */
        List<News> newsList = newsService.getNews(keyword);
        List<BertRequestDTO> requestDTO = toRequestDTO(newsList);

        BertResponseDTO bertResponseDTO = sendToBert(url, requestDTO);

        DailyIndex dailyIndex = toEntity(bertResponseDTO);
        dailyIndexRepository.save(dailyIndex);

        List<BertResult> individualResults = bertResponseDTO.getIndividualResults();
        for(BertResult b : individualResults) {
            News news = b.toNews();
            news.setDailyIndex(dailyIndex);
            newsRepository.save(news);
        }
    }

    private List<BertRequestDTO> toRequestDTO(List<News> news) {
        List<BertRequestDTO> requestDtos = new ArrayList<>();
        for(News n : news) {
            BertRequestDTO dto = new BertRequestDTO(n.getDate(), n.getContent());
            requestDtos.add(dto);
        }
        return requestDtos;
    }

    private DailyIndex toEntity(BertResponseDTO dto) {
        LocalDate now = LocalDate.now(ZoneId.of("Asia/Seoul"));
        return DailyIndex.builder()
                .date(now)
                .isPos(dto.getResult().equals("긍정"))
                .sentiment(dto.getPosProb())
                .build();
    }

    private BertResponseDTO sendToBert(String url, List<BertRequestDTO> requestDTO) {
        RestTemplate restTemplate = new RestTemplate();
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        Map<String, Object> request = new HashMap<>();
        request.put("news", requestDTO);

        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(request, headers);

        return restTemplate.postForEntity(
                url, entity, BertResponseDTO.class).getBody();
    }
}
